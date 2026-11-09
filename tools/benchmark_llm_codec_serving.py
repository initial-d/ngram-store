"""Fresh-worker original-key CPU lookup and cached complete Qwen serving."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .tree_node_quality import Reader, array_digest, digest, load, memory, write


def requests_for(common, split, count):
    metadata = load(common / 'metadata.json'); prepared = Path(metadata['prepared'])
    manifest = load(prepared / 'prepare.json')
    if digest(prepared / 'sequences.json') != manifest['payload_sha256']:
        raise ValueError('official sequences changed')
    payload = load(prepared / 'sequences.json'); requests = []
    for case in payload['cases']:
        if case['split'] != split: continue
        for group in ('rewrite', 'paraphrase', 'neighborhood'):
            for pair in case['groups'][group]:
                sequence = payload['sequences'][pair['new']]
                requests.append({'case_id': case['case_id'], 'group': group,
                    'ids': sequence['ids'][:sequence['target_start']]})
                if len(requests) == count: return requests
    raise ValueError('insufficient official request prefixes')


def batches_for(common, split, count, batch_size):
    """Distinct real unpadded prompts, grouped by exact token length."""
    if batch_size == 1:
        return [[row] for row in requests_for(common, split, count)]
    metadata = load(common / 'metadata.json'); prepared = Path(metadata['prepared'])
    manifest = load(prepared / 'prepare.json')
    if digest(prepared / 'sequences.json') != manifest['payload_sha256']:
        raise ValueError('official sequences changed')
    payload = load(prepared / 'sequences.json'); groups = {}; used = set(); batches = []
    for case in payload['cases']:
        if case['split'] != split: continue
        for group in ('rewrite', 'paraphrase', 'neighborhood'):
            for pair in case['groups'][group]:
                sequence = payload['sequences'][pair['new']]
                ids = sequence['ids'][:sequence['target_start']]; identity = tuple(ids)
                if identity in used: continue
                used.add(identity)
                bucket = groups.setdefault(len(ids), [])
                bucket.append({'case_id': case['case_id'], 'group': group, 'ids': ids})
                if len(bucket) == batch_size:
                    batches.append(list(bucket)); bucket.clear()
                    if len(batches) == count: return batches
    raise ValueError('insufficient distinct equally long official prefixes for real batching')


def cpu_worker(args):
    import faiss
    import torch
    from .codec_memory_adapter import CodecMultiHeadEmbedding
    from .llm_codec_frontier import original_row_digest
    from .tree_node_quality import environment
    torch.set_num_threads(1); torch.set_num_interop_threads(1); faiss.omp_set_num_threads(1)
    job = load(args.job); common = Path(job['common']); metadata = load(common / 'metadata.json')
    recipe = load(common / 'model-recipe.json')
    # Addressing uses exactly the trained compressed-token normalizer, without
    # loading a decoder or row matrix. Construct its pinned mapping directly.
    from .late_memory_replay import load_engram
    from .ngram_identity import Vocabulary, IdentityMapping
    module = load_engram(args.upstream)
    p = recipe['protocol']; vocab_sizes = p['variants'][recipe['variant']]['vocab_sizes']
    start = time.perf_counter(); before = memory()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(recipe['base_model'], local_files_only=True)
    config = load(Path(recipe['base_model']) / 'config.json')
    layer_id = config.get('text_config', config)['num_hidden_layers'] - p['suffix_layers']
    mapping = module.NgramHashMapping(engram_vocab_size=vocab_sizes, max_ngram_size=4,
        n_embed_per_ngram=64, n_head_per_ngram=2, layer_ids=[layer_id],
        tokenizer_name_or_path=recipe['base_model'], pad_id=tokenizer.pad_token_id,
        seed=recipe['seed'])
    vocabulary = Vocabulary.load(common / 'vocabulary.npz')
    mapping = IdentityMapping(mapping, vocabulary, True)
    heads = [n for order in mapping.vocab_size_across_layers[mapping.layer_ids[0]] for n in order]
    if heads != metadata['head_sizes']:
        raise ValueError('CPU original head layout changed')
    staged = CodecMultiHeadEmbedding(Reader(job, cache_bytes=args.cache_bytes), heads, 'cpu', profile=True, zero_sentinels=True)
    report = {'status': 'running', 'environment': environment(), 'phase_source_sha256':digest(__file__),
        'phase_source_file':'tools/benchmark_llm_codec_serving.py', 'job_sha256': digest(args.job),
        'cache_bytes': args.cache_bytes, 'rss_before_open': before,
        'load_seconds': time.perf_counter() - start, 'rss_after_open': memory(),
        'query': {}, 'row_only': {}, 'category': 'Official model tokens to original-key memory vectors; decoder excluded'}
    try:
        for split in ('calibration', 'test'):
            inputs = requests_for(common, split, args.requests)
            encoded = [np.asarray([row['ids']], dtype=np.int64) for row in inputs]
            report['query'][split] = {'requests': len(inputs), 'first_ms': [], 'warm_samples_ms': [], 'output_hashes': [], 'input_hashes': []}
            report['row_only'][split] = {'samples_ms': []}
            traces = []
            for ids in encoded:
                start = time.perf_counter_ns(); local = mapping.hash(ids)[mapping.layer_ids[0]]
                result = staged(torch.from_numpy(local)).numpy()
                report['query'][split]['first_ms'].append((time.perf_counter_ns() - start) / 1e6)
                report['query'][split]['output_hashes'].append(array_digest(result))
                report['query'][split]['input_hashes'].append(array_digest(ids)); traces.append(local)
            for _ in range(args.warmup):
                for ids in encoded: staged(torch.from_numpy(mapping.hash(ids)[mapping.layer_ids[0]]))
            for _ in range(args.repeat):
                for ids, local in zip(encoded, traces):
                    start = time.perf_counter_ns(); result = staged(torch.from_numpy(mapping.hash(ids)[mapping.layer_ids[0]]))
                    report['query'][split]['warm_samples_ms'].append((time.perf_counter_ns() - start) / 1e6)
                    start = time.perf_counter_ns(); staged(torch.from_numpy(local))
                    report['row_only'][split]['samples_ms'].append((time.perf_counter_ns() - start) / 1e6)
            for category in ('query', 'row_only'):
                samples = report[category][split].get('warm_samples_ms', report[category][split].get('samples_ms'))
                report[category][split].update(p50_ms=float(np.median(samples)), p95_ms=float(np.quantile(samples, .95)))
        report['rss_after_queries_before_integrity'] = memory()
        report['staging_after_queries_before_integrity'] = dict(staged.stats)
        rows_hash = original_row_digest(staged.reader, metadata['shape'][0])
        if rows_hash != load(job['report'])['decoded_row_sha256']:
            raise ValueError('CPU serving changed decoded original row bytes')
        report.update(status='passed', decoded_row_sha256=rows_hash, all_original_rows_verified=metadata['shape'][0],
            rss_after_query=memory(), staging_stats=staged.stats, source_row_bytes=metadata['shape'][0]*128,
            whole_serving_bytes=load(job['report'])['model_bytes']+(common/'metadata.json').stat().st_size,
            base_model_bytes=metadata['base_bytes'], host_loadavg=list(os.getloadavg()))
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        write(args.output, report); staged.close()


def gpu_worker(args):
    import faiss
    import torch
    from .benchmark_device_gather import gpu_snapshot, monitor_client_pid
    from .cached_memory_decode import cached_memory_session, generate_cached
    from .counterfact_native import restore, restore_independent_control, digest as tensor_digest
    from .llm_codec_frontier import reopen_codec, original_row_digest
    from .tree_node_quality import environment
    torch.set_num_threads(1); torch.set_num_interop_threads(1); faiss.omp_set_num_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    job = load(args.job); common = Path(job['common']); metadata = load(common / 'metadata.json')
    recipe = load(common / 'model-recipe.json'); selector = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not selector or ',' in selector or torch.cuda.device_count() != 1:
        raise ValueError('exactly one selected CUDA device required')
    uuid = subprocess.check_output(['nvidia-smi', '--id='+selector, '--query-gpu=uuid', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    initial = gpu_snapshot(uuid, own_pid=0)
    if initial['external_compute_pids'] or initial['utilization_percent'] > 5:
        raise ValueError('GPU competition before model loading')
    probe = torch.empty(1, device='cuda'); torch.cuda.synchronize()
    monitor_pid = monitor_client_pid(gpu_snapshot(uuid, own_pid=0)); del probe
    runtime_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not runtime_uuid.startswith('GPU-'): runtime_uuid = 'GPU-' + runtime_uuid
    if runtime_uuid != uuid: raise ValueError('CUDA and NVML selected different GPUs')
    report = {'status': 'running', 'environment': {**environment(), 'device':'cuda', 'gpu':torch.cuda.get_device_name(0)},
        'phase_source_sha256':digest(__file__), 'cached_decode_source_sha256':digest(Path(__file__).with_name('cached_memory_decode.py')),
        'codec_adapter_source_sha256':digest(Path(__file__).with_name('codec_memory_adapter.py')),
        'job_sha256': digest(args.job), 'mode': args.mode, 'gpu_before':initial, 'gpu_samples':[],
        'cache_bytes':args.cache_bytes, 'batch_size':args.batch_size, 'decode_steps':args.decode_steps,
        'selected_gpu_selector':selector,
        'batch_contract':'Each batch uses distinct real official prompts with equal token length; no padding, repetition or position remapping',
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'rss_before_open':memory(), 'requests':[], 'samples':[],
        'contract':'Actual complete prefill plus native decoder KV/recurrent cache; causal TinyEngram history, original IDs and unchanged nonrow weights',
        'rounding_contract':'Cache vs full-prefix differences are measured separately. Source and byte-exact codecs execute the same cached schedule.'}
    model = replay = staged = None
    def snapshot():
        value = gpu_snapshot(uuid, own_pid=monitor_pid); report['gpu_samples'].append(value)
        if value['external_compute_pids'] or monitor_pid not in value['compute_pids']:
            raise ValueError('GPU competition appeared during serving')
    try:
        started = time.perf_counter()
        if args.mode == 'source':
            _, model, replay, state = restore(Path(metadata['source_run']), Path(recipe['base_model']), args.upstream)
            del state
        elif args.mode == 'independent':
            control_report, model, replay = restore_independent_control(args.control_run, args.upstream)
            if (control_report['prepared_sha256'] != recipe['prepared_sha256']
                    or control_report['seed'] != recipe['seed']
                    or control_report['frozen_weight_sha256'] != recipe['frozen_weight_sha256']):
                raise ValueError('independent control uses a different source/data')
            report['independent_control_report_sha256'] = digest(args.control_run / 'report.json')
            report['independent_control_delta_bytes'] = control_report['delta']['bytes']
        else:
            model, replay, staged = reopen_codec(job, args.upstream, cache_bytes=args.cache_bytes)
        gc.collect(); torch.cuda.synchronize()
        report.update(open_seconds=time.perf_counter()-started, rss_after_open=memory(),
            cuda_after_open={'allocated':torch.cuda.memory_allocated(), 'reserved':torch.cuda.memory_reserved(),
                'peak_allocated':torch.cuda.max_memory_allocated()})
        requests = batches_for(common, 'test', args.requests, args.batch_size)
        prefixes = [torch.tensor([row['ids'] for row in batch], dtype=torch.int64) for batch in requests]
        report['unique_prompt_count'] = len({tuple(row['ids']) for batch in requests for row in batch})
        continuations = []
        with torch.no_grad():
            for request, prefix in zip(requests, prefixes):
                snapshot(); torch.cuda.synchronize(); start = time.perf_counter_ns()
                generated, logits = generate_cached(replay, prefix, steps=args.decode_steps, global_step=recipe['global_step'])
                torch.cuda.synchronize(); first = (time.perf_counter_ns()-start)/1e6
                full = replay.complete(prefix, True, 1).cpu()
                error = float((full-logits[0]).abs().max())
                # Cache arithmetic can round differently. Pair source/codec
                # cached hashes, and expose the comparison to ordinary forwards.
                continuations.append(generated)
                report['requests'].append({'case_id':request[0]['case_id'], 'group':request[0]['group'],
                    'batch_case_ids':[row['case_id'] for row in request], 'batch_groups':[row['group'] for row in request],
                    'prompt_hashes':[array_digest(np.asarray(row['ids'],dtype='<i8')) for row in request],
                    'input_ids_sha256':tensor_digest(prefix), 'input_tokens':prefix.shape[1],
                    'cached_logit_hashes':[tensor_digest(x) for x in logits],
                    'generated_ids_sha256':tensor_digest(generated), 'first_complete_ms':first,
                    'cached_vs_full_prefill_max_abs':error})
                snapshot()
            for _ in range(args.warmup):
                for prefix, continuation in zip(prefixes, continuations):
                    generate_cached(replay, prefix, steps=args.decode_steps, global_step=recipe['global_step'], continuation=continuation)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeat):
                for prefix, continuation in zip(prefixes, continuations):
                    snapshot(); prefill_start=time.perf_counter_ns(); torch.cuda.synchronize()
                    with cached_memory_session(replay, recipe['global_step']) as cache:
                        with replay.context(prefix): result=model(input_ids=prefix.cuda(), use_cache=True, logits_to_keep=1)
                        torch.cuda.synchronize(); prefill_ms=(time.perf_counter_ns()-prefill_start)/1e6
                        full_ids=prefix; decode_ms=[]
                        for step in range(args.decode_steps):
                            token=continuation[:,step]; full_ids=torch.cat((full_ids,token[:,None]),dim=1)
                            torch.cuda.synchronize(); start=time.perf_counter_ns()
                            with replay.context(full_ids):
                                result=model(input_ids=token[:,None].cuda(), past_key_values=result.past_key_values,
                                    use_cache=True, logits_to_keep=1)
                            torch.cuda.synchronize(); decode_ms.append((time.perf_counter_ns()-start)/1e6)
                        report['samples'].append({'prefill_ms':prefill_ms, 'decode_ms':decode_ms,
                            'memory_history_tokens':getattr(cache, 'history_length', 0)})
                    snapshot()
        report['rss_after_queries_before_integrity'] = memory()
        if staged is not None:
            staged.profile=True
            generate_cached(replay, prefixes[0], steps=args.decode_steps, global_step=recipe['global_step'], continuation=continuations[0])
            report['staging_stats']=staged.stats
            report['rss_after_staging_probe_before_integrity'] = memory()
            row_hash=original_row_digest(staged.reader,metadata['shape'][0])
            if row_hash!=load(job['report'])['decoded_row_sha256']:raise ValueError('GPU worker original row decode changed')
            report['decoded_row_sha256']=row_hash
        snapshot()
        prefill=[s['prefill_ms'] for s in report['samples']]; decoding=[v for s in report['samples'] for v in s['decode_ms']]
        report.update(status='passed', prefill_p50_ms=float(np.median(prefill)), prefill_p95_ms=float(np.quantile(prefill,.95)),
            decode_p50_ms=float(np.median(decoding)),decode_p95_ms=float(np.quantile(decoding,.95)),
            decode_tokens_per_second=args.batch_size*1000/float(np.mean(decoding)),
            cuda_after_query={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()},
            rss_after_query=memory(),host_loadavg=list(os.getloadavg()),source_row_bytes=metadata['shape'][0]*128,
            complete_serving_bytes=load(job['report'])['model_bytes']+(common/'metadata.json').stat().st_size,
            base_model_bytes=metadata['base_bytes'])
        if args.mode == 'independent':
            report.update(source_row_bytes=control_report['identity']['row_bytes'],
                complete_serving_bytes=control_report['delta']['bytes']+(args.control_run/'report.json').stat().st_size,
                independent_registry_required=False,
                independent_serving_files=['delta.safetensors','report.json'])
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        write(args.output,report)
        if staged is not None:staged.close()
        if replay is not None:replay.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('cpu','gpu'))
    for name in ('job','output','upstream'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=('source','codec','independent'),default='codec')
    parser.add_argument('--control-run',type=Path)
    parser.add_argument('--cache-bytes',type=int,default=0)
    parser.add_argument('--requests',type=int,default=32)
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--decode-steps',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=1)
    args=parser.parse_args()
    if args.output.exists() or args.requests<1 or args.repeat<1 or args.warmup<0 or args.cache_bytes<0:
        parser.error('fresh output and valid positive budgets required')
    if args.mode == 'independent' and args.control_run is None: parser.error('independent mode requires a trained control run')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.phase=='cpu':cpu_worker(args)
    else:gpu_worker(args)


if __name__=='__main__':main()
