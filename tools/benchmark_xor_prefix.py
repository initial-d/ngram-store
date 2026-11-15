"""Paired CPU gathers of existing trained codes on saved original-ID traces.

No training, repacking or GPU work. Both libraries read the same immutable
table; every result is compared with its original code bytes. This measures
row lookup after addressing, not quantizer decoding or complete LLM inference.
"""
import argparse
import ctypes
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

from .native_row_store import NativeRows


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def benchmark(args):
    if args.repeat < 1 or args.rounds < 1 or args.warmup < 0 or args.cache_bytes < 0:
        raise ValueError('positive rounds/repeats and nonnegative warmup/cache required')
    if args.core is not None:
        os.sched_setaffinity(0, {args.core})
    trace_metadata = json.loads(args.traces.with_suffix('.json').read_text())
    if digest(args.traces) != trace_metadata['trace_sha256']:
        raise ValueError('original trace changed')
    source = np.load(args.codes, mmap_mode='r', allow_pickle=False)
    trace = np.load(args.traces, allow_pickle=False)
    queries = []
    for i, record in enumerate(trace_metadata['requests']):
        ids = trace['ids' + str(i)]
        if ids.dtype.kind not in 'iu' or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(source)):
            raise ValueError('invalid original row IDs')
        if hashlib.sha256(ids.tobytes()).hexdigest() != record['ids_sha256']:
            raise ValueError('original query changed')
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        expected = source[ids].tobytes()
        queries.append((ids, expected))
    width = int(source.shape[1] * source.dtype.itemsize)
    result = {'commit': args.commit, 'command': shlex.join([sys.executable, *sys.argv]),
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'loadavg': list(os.getloadavg()), 'device': 'cpu', 'gpu': 'unused',
        'rows': len(source), 'row_bytes': width, 'repeat': args.repeat, 'rounds': args.rounds,
        'warmup': args.warmup, 'cache_bytes': args.cache_bytes, 'cache_policy': args.cache_policy,
        'trace_sha256': digest(args.traces), 'trace_metadata_sha256': digest(args.traces.with_suffix('.json')),
        'table_sha256': digest(args.table), 'codes_sha256': digest(args.codes),
        'libraries_sha256': {name: digest(path) for name, path in
                               (('baseline', args.baseline_library), ('candidate', args.library))},
        'source_sha256': digest(__file__), 'round_results': [],
        'scope': 'Saved original canonical row queries on trained Qwen memory codes. Warm OS cache, single CPU core; paired rounds are descriptive, not confidence bounds or complete inference speedups. Original vectors, source codes and query IDs are not exported.',
        'cost': 'Existing CPU resources; no GPU, training or paid API. Energy/monetary costs unmeasured.'}
    try:
        result['torch'] = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        result['torch'] = 'unused, not installed'
    cpuinfo = Path('/proc/cpuinfo')
    result['cpu'] = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith('model name')), platform.machine()) if cpuinfo.exists() else platform.machine()
    for round_id in range(args.rounds):
        with NativeRows(args.table, args.baseline_library, cache_bytes=args.cache_bytes,
                        cache_policy=args.cache_policy) as baseline, NativeRows(
                        args.table, args.library, cache_bytes=args.cache_bytes,
                        cache_policy=args.cache_policy) as candidate:
            stores = {'baseline': baseline, 'candidate': candidate}
            if any(s.header != {'n_rows': len(source), 'row_bytes': width} for s in stores.values()):
                raise ValueError('table/source dimensions differ')
            output = {name: [np.empty((len(ids), width), dtype=np.uint8) for ids, _ in queries]
                      for name in stores}
            counts = {name: {key: 0 for key in ('blocks', 'decodes', 'compressed_bytes', 'decoded_bytes', 'xor_bytes')}
                      for name in stores}
            samples = {name: [] for name in stores}
            for repeat in range(-args.warmup, args.repeat):
                for i, (ids, expected) in enumerate(queries):
                    order = ('baseline', 'candidate') if (round_id + repeat + i) % 2 else ('candidate', 'baseline')
                    work = {}
                    for name in order:
                        store = stores[name]; out = output[name][i]
                        started = time.perf_counter_ns()
                        store.gather_into(ids.ctypes.data, len(ids), out.ctypes.data, out.nbytes)
                        elapsed = time.perf_counter_ns() - started
                        if out.tobytes() != expected:
                            raise ValueError('original code bytes changed: ' + name)
                        work[name] = (store.last_block_reads, store.last_decompressions,
                            store.last_compressed_bytes, store.last_decoded_bytes, store.last_xor_bytes)
                        if repeat >= 0:
                            samples[name].append(elapsed / 1000.)
                            for key, count in zip(counts[name], work[name]):
                                counts[name][key] += count
                    if work['baseline'][:4] != work['candidate'][:4]:
                        raise ValueError('stream decode/cache work differs')
            result['round_results'].append({'round': round_id, 'samples_us': samples, 'work': counts,
                'p50_us': {name: statistics.median(values) for name, values in samples.items()},
                'paired_geometric_speed_ratio': math.exp(statistics.mean(math.log(a / b) for a, b in
                    zip(samples['baseline'], samples['candidate']))), 'original_code_bytes_match': True})
    result['status'] = 'passed'
    result['geometric_speed_ratio'] = math.exp(statistics.mean(math.log(r['paired_geometric_speed_ratio'])
                                                              for r in result['round_results']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'speed_ratio': result['geometric_speed_ratio']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('table', 'codes', 'traces', 'baseline-library', 'library', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--core', type=int)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--cache-bytes', type=int, default=0)
    parser.add_argument('--cache-policy', choices=('blocks', 'rows'), default='blocks')
    benchmark(parser.parse_args())


if __name__ == '__main__':
    main()
