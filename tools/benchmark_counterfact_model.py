"""Separate-process real Qwen source/native startup and complete-prefix checks.

Requires an idle selected GPU for CUDA timing. No training, cached-prefix replay,
KV cache, approximate weights, or generated-answer metrics. Original next-token
logit hashes are paired by official case/group/prefix across worker reports.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .benchmark_counterfact_storage import environment, write, rss
from .benchmark_device_gather import gpu_snapshot, monitor_client_pid
from .counterfact_native import restore, reopen, digest
from .engram_row_adapter import file_sha256


def worker(args):
    env = {**environment(), 'device': args.device,
           'gpu': 'selected CUDA device; details recorded after initialization' if args.device == 'cuda' else 'unused'}
    report = {'status': 'running', 'mode': args.mode, 'environment': env,
              'execution_device': args.device, 'warmup': args.warmup, 'repeat': args.repeat,
              'rss_before_open': rss(), 'requests': [],
              'limitations': ['Fixed official test prefixes; next-token FP32 logits, not full MultiCounterFact metrics or generation.',
                  'Batch one, eager attention, complete uncached forwards; no concurrent serving, KV cache or padding.',
                  'Checksums/frozen-weight validation warm filesystem and contribute to startup time/peak RSS.',
                  'Model base dominates these roughly 74 MB memory tables; no billion-row capacity extrapolation.',
                  'Existing shared host; no paid APIs or new training. Monetary/energy cost unmeasured.']}
    replay, staged = None, None
    uuid, monitor_pid = None, None
    try:
        if args.device == 'cuda':
            selector = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if not selector or ',' in selector:
                raise ValueError('expose exactly one selected GPU')
            uuid = subprocess.check_output(['nvidia-smi', '--id=' + selector,
                '--query-gpu=uuid', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
            report['gpu_before'] = gpu_snapshot(uuid, own_pid=0)
            if report['gpu_before']['external_compute_pids'] or report['gpu_before']['utilization_percent'] > 5:
                raise ValueError('selected GPU is contended; measurement blocked')
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError('one CUDA device required')
            torch.cuda.set_per_process_memory_fraction(.65)
            probe = torch.empty(1, device='cuda')
            torch.cuda.synchronize()
            runtime_uuid = str(torch.cuda.get_device_properties(0).uuid)
            runtime_uuid = runtime_uuid if runtime_uuid.startswith('GPU-') else 'GPU-' + runtime_uuid
            if runtime_uuid != uuid:
                raise ValueError('CUDA runtime and monitoring selected different GPUs')
            monitor_pid = monitor_client_pid(gpu_snapshot(uuid, own_pid=0))
            del probe
            report.update(cuda_runtime=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
                          monitor_pid=monitor_pid, worker_namespace_pid=os.getpid())
            report['environment']['gpu'] = report['gpu_name']
            torch.cuda.reset_peak_memory_stats()
        synchronize = torch.cuda.synchronize if args.device == 'cuda' else lambda: None
        started = time.perf_counter()
        source_report = json.loads((args.run / 'report.json').read_text())
        if args.mode == 'source':
            _, model, replay, state = restore(args.run, args.base, args.upstream, device=args.device)
            del state
        else:
            model, replay, staged = reopen(args.bundle, args.upstream, args.native_library, device=args.device)
            recipe = json.loads((args.bundle / 'model-recipe.json').read_text())
            for key in ('seed', 'variant', 'protocol', 'identity', 'frozen_weight_sha256', 'frozen_decoder_suffix_sha256'):
                if recipe[key] != source_report[key]:
                    raise ValueError('native bundle differs from source run: ' + key)
            if str(args.base) != recipe['base_model']:
                raise ValueError('native base path differs')
        gc.collect()
        synchronize()
        report.update(open_seconds=time.perf_counter() - started, rss_after_open=rss(),
            source_run_report_sha256=file_sha256(args.run / 'report.json'),
            source_delta_sha256=source_report['delta']['sha256'],
            source_row_bytes=source_report['identity']['row_bytes'],
            memory_row_parameter_present=any(n == 'ngram_memory.multi_head_embedding.embedding.weight' for n, _ in model.named_parameters()))
        if args.device == 'cuda':
            report['cuda_after_open'] = {'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        prepared = json.loads((args.prepared / 'prepare.json').read_text())
        if file_sha256(args.prepared / 'prepare.json') != source_report['prepared_sha256'] or file_sha256(args.prepared / 'sequences.json') != prepared['payload_sha256']:
            raise ValueError('source evaluation inputs changed')
        payload = json.loads((args.prepared / 'sequences.json').read_text())
        requests = []
        for c in payload['cases']:
            if c['split'] != 'test':
                continue
            for group in ('rewrite', 'paraphrase', 'neighborhood'):
                for pair in c['groups'][group]:
                    seq = payload['sequences'][pair['new']]
                    ids = torch.tensor([seq['ids'][:seq['target_start']]], dtype=torch.int64)
                    requests.append((c['case_id'], group, ids))
                    if len(requests) == args.requests:
                        break
                if len(requests) == args.requests:
                    break
            if len(requests) == args.requests:
                break
        if len(requests) != args.requests:
            raise ValueError('insufficient official test prefixes')
        del payload
        gc.collect()
        with torch.no_grad():
            for case, group, ids in requests:
                output = replay.complete(ids, True, 1)
                synchronize()
                report['requests'].append({'case_id': case, 'group': group, 'input_tokens': ids.shape[1],
                    'input_ids_sha256': digest(ids), 'next_token_logits_sha256': digest(output)})
                del output
            for _ in range(args.warmup):
                for _, _, ids in requests:
                    output = replay.complete(ids, True, 1)
                    synchronize()
                    del output
            if uuid:
                snapshot = gpu_snapshot(uuid, own_pid=monitor_pid)
                report['gpu_before_timing'] = snapshot
                if snapshot['external_compute_pids'] or monitor_pid not in snapshot['compute_pids']:
                    raise ValueError('GPU competition appeared before timing')
                torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(args.repeat):
                for _, _, ids in requests:
                    synchronize()
                    started = time.perf_counter_ns()
                    output = replay.complete(ids, True, 1)
                    synchronize()
                    samples.append((time.perf_counter_ns() - started) / 1e6)
                    del output
            if uuid:
                report['gpu_after_timing'] = gpu_snapshot(uuid, own_pid=monitor_pid)
                if report['gpu_after_timing']['external_compute_pids'] or monitor_pid not in report['gpu_after_timing']['compute_pids']:
                    raise ValueError('GPU competition appeared during timing')
                report['cuda_after_query'] = {'allocated_bytes': torch.cuda.memory_allocated(),
                    'reserved_bytes': torch.cuda.memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        report.update(status='passed', samples_ms=samples, p50_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)), requests_per_second=len(samples)/(sum(samples)/1000),
            rss_after_query=rss(), staging_stats=staged.stats if staged else None)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output, report)
        if staged is not None:
            staged.close()
        if replay is not None:
            replay.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'base', 'upstream', 'prepared', 'native-library', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--mode', choices=('source', 'native'), required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--requests', type=int, default=64)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeat', type=int, default=3)
    args = p.parse_args()
    if args.mode == 'native' and args.bundle is None:
        p.error('native mode requires a bundle')
    if args.output.exists() or not 1 <= args.requests <= 1024 or args.repeat < 1 or args.warmup < 0:
        p.error('fresh output, 1..1024 requests, positive repeat and nonnegative warmup required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    worker(args)


if __name__ == '__main__':
    main()
