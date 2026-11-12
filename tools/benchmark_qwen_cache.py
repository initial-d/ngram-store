"""Real canonical Qwen row traces under bounded caches and process concurrency.

This measures canonical-row gather after token addressing, not complete model
inference. Worker handles are independent; shared files use the OS page cache.
No global cache drops or machine-wide pressure allocations are performed.
"""
import argparse
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .benchmark_large_application import memory_details
from .tree_node_quality import Reader, array_digest, digest, environment, load, write


def traces(args):
    from .benchmark_llm_codec_serving import requests_for
    from .late_memory_replay import load_engram
    from .ngram_identity import Vocabulary, IdentityMapping
    from transformers import AutoTokenizer
    cfg = load(args.prepared); first = load(cfg['entries'][0]['job'])
    common = Path(first['common']); recipe = load(common / 'model-recipe.json')
    module = load_engram(args.upstream); protocol = recipe['protocol']
    tokenizer = AutoTokenizer.from_pretrained(recipe['base_model'], local_files_only=True)
    config = load(Path(recipe['base_model']) / 'config.json')
    layer = config.get('text_config', config)['num_hidden_layers'] - protocol['suffix_layers']
    source = module.NgramHashMapping(engram_vocab_size=protocol['variants'][recipe['variant']]['vocab_sizes'],
        max_ngram_size=4, n_embed_per_ngram=64, n_head_per_ngram=2, layer_ids=[layer],
        tokenizer_name_or_path=recipe['base_model'], pad_id=tokenizer.pad_token_id, seed=recipe['seed'])
    mapping = IdentityMapping(source, Vocabulary.load(common / 'vocabulary.npz'), True)
    sizes = load(common / 'metadata.json')['head_sizes']
    offsets = np.cumsum([0, *sizes[:-1]], dtype=np.int64)
    arrays = {}; reports = []
    for i, row in enumerate(requests_for(common, 'test', 32)):
        local = mapping.hash(np.asarray([row['ids']], dtype=np.int64))[layer]
        ids = (local + offsets).reshape(-1).astype('<i8')
        arrays[f'ids{i}'] = ids; arrays[f'zero{i}'] = (local == 0).reshape(-1)
        reports.append({'case_id': row['case_id'], 'group': row['group'], 'ids_sha256': array_digest(ids)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    write(args.output.with_suffix('.json'), {'prepared_sha256': digest(args.prepared), 'trace_sha256': digest(args.output),
        'requests': reports, 'source_sha256': digest(__file__), 'domain': 'Official test token prefixes, original canonical row IDs, no key replication.'})


def actor(command):
    import faiss
    path, trace, budget, policy, core, repeat, barrier = command
    os.sched_setaffinity(0, {core}); faiss.omp_set_num_threads(1)
    job = load(path); payload = np.load(trace); reader = Reader(job, cache_bytes=budget)
    native = getattr(reader.reader, 'native', None)
    if native is not None: native.set_cache_policy(policy)
    count = len(payload.files) // 2
    output_hashes = []; first_ms = []; before = memory_details()
    def request(i):
        ids = payload[f'ids{i}']
        if reader.mapping is not None: ids = reader.mapping[ids]
        values = reader.rows(ids); values[payload[f'zero{i}']] = 0
        return values
    try:
        for i in range(count):
            start = time.perf_counter_ns(); values = request(i)
            first_ms.append((time.perf_counter_ns()-start)/1e6); output_hashes.append(array_digest(values))
        for _ in range(2):
            for i in range(count): request(i)
        barrier.wait(timeout=120)
        start_wall = time.monotonic_ns(); samples = []; counters = {'decompressions': 0, 'block_reads': 0, 'cache_hits': 0}
        for _ in range(repeat):
            for i in range(count):
                start = time.perf_counter_ns(); request(i); samples.append((time.perf_counter_ns()-start)/1e6)
                if native is not None:
                    for key in counters: counters[key] += getattr(native, 'last_' + key)
        end_wall = time.monotonic_ns()
        return {'status': 'passed', 'pid': os.getpid(), 'core': core, 'start_ns': start_wall, 'end_ns': end_wall,
            'completed_requests': count*repeat, 'samples_ms': samples, 'first_new_handle_ms': first_ms,
            'output_hashes': output_hashes, 'before': before, 'after': memory_details(), 'work': counters,
            'requested_cache_bytes': budget, 'actual_native_cache_bytes': native.cache_bytes if native else 0,
            'native_decoder': native.decoder_name if native else None, 'policy': policy if native else 'none'}
    finally: reader.close(); payload.close()


def lane(args):
    cfg = load(args.prepared); selected = load(args.selection)['selected']
    entries = []
    for choice in selected:
        codec = choice['codec']
        for e in cfg['entries']:
            if e['codec'] == codec and (e['variant'] == 'direct' or
                (e.get('block_rows'), e.get('predictor')) in
                ((choice['block_rows'], choice['predictor']), (256, 'xor'))):
                entries.append(e)
    reference_hashes = {}
    for entry in entries:
        name = Path(entry['job']).stem
        for budget, policy in ((0, 'blocks'), (1<<20, 'blocks'), (8<<20, 'blocks'), (1<<20, 'rows'), (8<<20, 'rows')):
            for workers in (1, 2, 4, 8):
                out = args.output / f'{name}-{budget}-{policy}-w{workers}.json'
                if out.exists() and load(out)['status'] == 'passed':
                    report = load(out)
                else:
                    ctx = multiprocessing.get_context('spawn')
                    with ctx.Manager() as manager:
                        barrier = manager.Barrier(workers)
                        commands = [(entry['job'], str(args.traces), budget, policy, args.core_base + i, 5, barrier)
                                    for i in range(workers)]
                        with ctx.Pool(workers) as pool: results = pool.map(actor, commands)
                    all_samples = [s for r in results for s in r['samples_ms']]
                    elapsed = (max(r['end_ns'] for r in results) - min(r['start_ns'] for r in results)) / 1e9
                    report = {'status': 'passed', 'environment': environment(), 'entry': entry,
                        'source_sha256': digest(__file__), 'job_sha256': digest(entry['job']), 'trace_sha256': digest(args.traces),
                        'workers': results, 'concurrency': workers, 'elapsed_wall_seconds': elapsed,
                        'requests_per_second': sum(r['completed_requests'] for r in results)/elapsed,
                        'p50_ms': float(np.median(all_samples)), 'p95_ms': float(np.quantile(all_samples,.95)),
                        'p99_ms': float(np.quantile(all_samples,.99)), 'total_native_cache_bytes': sum(r['actual_native_cache_bytes'] for r in results),
                        'scope': 'Canonical-row query only. New independent handles, warm OS pages, no token hash cost or GPU model forward. Per-worker budgets include native cache metadata; file pages and interpreter RSS are separate.'}
                    write(out, report)
                hashes = report['workers'][0]['output_hashes']
                if any(r['output_hashes'] != hashes for r in report['workers']): raise ValueError('concurrent readers changed vectors')
                if entry['variant'] == 'direct': reference_hashes[entry['codec']] = hashes
                elif reference_hashes[entry['codec']] != hashes: raise ValueError('cache/reader layout changed actual query vectors')
    write(args.output / 'complete.json', {'status': 'complete', 'reports': {str(p): digest(p) for p in args.output.glob('*.json') if p.name != 'complete.json'},
        'prepared_sha256': digest(args.prepared), 'selection_sha256': digest(args.selection), 'source_sha256': digest(__file__)})


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='phase', required=True)
    c = sub.add_parser('traces')
    for name in ('prepared', 'upstream', 'output'): c.add_argument('--' + name, type=Path, required=True)
    c = sub.add_parser('lane')
    for name in ('prepared', 'selection', 'traces', 'output'): c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--core-base', type=int, required=True)
    args = p.parse_args(); {'traces': traces, 'lane': lane}[args.phase](args)


if __name__ == '__main__': main()
