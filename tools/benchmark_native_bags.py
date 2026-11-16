"""Paired NumPy/C bag calls on saved original-ID traces and frozen FP32 codes."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys
import time

import numpy as np

from .benchmark_native_gather import digest
from .native_row_store import NativeRows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('table', 'codes', 'traces', 'baseline-library', 'library', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--core', type=int)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 32])
    parser.add_argument('--mode', choices=('sum', 'mean'), default='mean')
    parser.add_argument('--chunk-rows', type=int, default=4096)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    if (min(args.batch_sizes + [args.chunk_rows, args.repeat, args.rounds]) < 1 or args.warmup < 0):
        raise ValueError('positive sizes/repeats and nonnegative warmup required')
    if args.core is not None:
        os.sched_setaffinity(0, {args.core})
    metadata = json.loads(args.traces.with_suffix('.json').read_text())
    if digest(args.traces) != metadata['trace_sha256']:
        raise ValueError('original trace changed')
    codes = np.load(args.codes, mmap_mode='r', allow_pickle=False)
    if codes.ndim != 2 or codes.dtype != np.dtype('<f4'):
        raise ValueError('frozen little-endian FP32 codes required')
    traces = np.load(args.traces, allow_pickle=False)
    docs, features = [], []
    for i, record in enumerate(metadata['requests']):
        ids = traces['ids' + str(i)]
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or
                (len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= len(codes)))):
            raise ValueError('invalid original IDs')
        if hashlib.sha256(ids.tobytes()).hexdigest() != record['ids_sha256']:
            raise ValueError('original query changed')
        docs.append(np.ascontiguousarray(ids, dtype=np.uint64))
        values = codes[ids].astype(np.float64).sum(axis=0)
        if args.mode == 'mean':
            values /= max(len(ids), 1)
        features.append(values.astype(np.float32))
    if not docs:
        raise ValueError('nonempty trace required')
    report = {'commit': args.commit, 'command': shlex.join([sys.executable, *sys.argv]),
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'loadavg': list(os.getloadavg()),
        'cpu': next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), 'device': 'cpu', 'gpu': 'unused',
        'scope': 'Paired public NumPy bag calls including validation and C reduction; warm file cache, zero native cache. Excludes tokenization, classifier head and model inference. Ratios are descriptive, not confidence intervals.',
        'cost': 'Existing CPU resources; no GPU, training or paid API. Monetary/energy cost unmeasured.',
        'trace_domain': metadata['domain'], 'rows': len(codes), 'dim': codes.shape[1],
        'row_bytes': codes.shape[1] * 4, 'chunk_rows': args.chunk_rows, 'mode': args.mode,
        'repeat': args.repeat, 'rounds': args.rounds, 'warmup': args.warmup,
        'requests': len(docs), 'rows_per_document': [len(ids) for ids in docs],
        'hashes': {name: digest(path) for name, path in
            [('table', args.table), ('codes', args.codes), ('traces', args.traces),
             ('trace_metadata', args.traces.with_suffix('.json')), ('baseline_library', args.baseline_library),
             ('candidate_library', args.library), ('benchmark_source', Path(__file__))]}, 'results': []}
    try:
        report['torch'] = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        report['torch'] = 'unused, not installed'
    for batch in args.batch_sizes:
        requests = []
        for begin in range(0, len(docs), batch):
            selected = docs[begin:begin + batch]
            ids = np.concatenate(selected)
            offsets = np.r_[0, np.cumsum([len(x) for x in selected], dtype=np.uint64)].astype(np.uint64)
            expected = np.ascontiguousarray(features[begin:begin + batch])
            requests.append((ids, offsets, expected))
        case = {'batch_size': batch, 'round_results': []}
        for round_id in range(args.rounds):
            with NativeRows(args.table, args.baseline_library) as baseline, NativeRows(args.table, args.library) as candidate:
                stores = {'baseline': baseline, 'candidate': candidate}
                if any(s.header != {'n_rows': len(codes), 'row_bytes': codes.shape[1] * 4} for s in stores.values()):
                    raise ValueError('table/source shape mismatch')
                report['decoders'] = {name: store.decoder_name for name, store in stores.items()}
                output = {name: [np.empty_like(expected) for _, _, expected in requests] for name in stores}
                samples = {name: [] for name in stores}
                stage = {name: 0 for name in stores}
                work = {name: [0] * 6 for name in stores}
                for repeat in range(-args.warmup, args.repeat):
                    for i, (ids, offsets, expected) in enumerate(requests):
                        counts = {}
                        order = ('baseline', 'candidate') if (round_id + repeat + i) % 2 else ('candidate', 'baseline')
                        for name in order:
                            store = stores[name]; out = output[name][i]
                            started = time.perf_counter_ns()
                            store.embedding_bag_array(ids, offsets, mode=args.mode, chunk_rows=args.chunk_rows, out=out)
                            elapsed = (time.perf_counter_ns() - started) / 1000.
                            if out.tobytes() != expected.tobytes():
                                raise ValueError('FP32 feature bytes differ from source FP64 reference: ' + name)
                            counts[name] = [store.last_block_reads, store.last_decompressions,
                                store.last_cache_hits, store.last_row_cache_hits, store.last_decoded_bytes, store.last_xor_bytes]
                            if repeat >= 0:
                                samples[name].append(elapsed)
                                stage[name] = max(stage[name], store.last_bag_staging_bytes)
                                work[name] = [a + b for a, b in zip(work[name], counts[name])]
                        if counts['baseline'] != counts['candidate']:
                            raise ValueError('logical read/decode work changed')
                ratio = math.exp(statistics.mean(math.log(a / b) for a, b in zip(samples['baseline'], samples['candidate'])))
                case['round_results'].append({'round': round_id, 'samples_us': samples,
                    'p50_us': {name: statistics.median(values) for name, values in samples.items()},
                    'p95_us': {name: sorted(values)[math.ceil(.95 * len(values)) - 1] for name, values in samples.items()},
                    'paired_geometric_speed_ratio': ratio, 'maximum_row_staging_bytes': stage,
                    'work_fields': ['blocks','decodes','cache_hits','row_cache_hits','decoded_bytes','xor_bytes'],
                    'work': work, 'source_fp64_to_fp32_features_byte_match': True})
        case['geometric_speed_ratio'] = math.exp(statistics.mean(math.log(r['paired_geometric_speed_ratio']) for r in case['round_results']))
        report['results'].append(case)
    report['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'status': 'passed',
        'speed_ratios': {r['batch_size']: r['geometric_speed_ratio'] for r in report['results']}}), flush=True)


if __name__ == '__main__':
    main()
