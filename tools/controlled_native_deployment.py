"""Prepare immutable native layout controls and run balanced Qwen serving lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

from .tree_node_quality import digest, load, serving_files, write


def prepare_model(args, source):
    from .faiss_row_codec import fit_codec, pack_codec_native
    import numpy as np
    output = args.output / source.name
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for codec in args.codecs.split(','):
        original = source / 'jobs' / ('faiss-' + codec + '.json')
        if codec == 'fp32' and not original.exists():
            # Original trained rows, not a new fit or repeated capacity fixture.
            direct = output / 'direct-fp32'
            if not direct.exists():
                rows = np.load(source / 'common/source-rows.npy', mmap_mode='r')
                fit_codec(direct, rows, 'fp32', rows[:1])
            old = load(source / 'jobs/raw-fp32.json')
            old_report = load(old['report'])
            job = {**old, 'backend': 'faiss', 'directory': str(direct), 'encoding': 'fp32', 'name': 'direct-fp32'}
        else:
            job = load(original); old_report = load(job['report']); direct = Path(job['directory'])
        current_direct = {**job, 'native_library': str(args.accelerated_library)}
        direct_job = output / 'jobs' / ('direct-' + codec + '.json')
        write(direct_job, current_direct)
        entries.append({'codec': codec, 'variant': 'direct', 'job': str(direct_job)})
        for block_rows in (16, 64, 256, 1024):
            for predictor in ('none', 'xor'):
                name = f'{codec}-rows{block_rows}-{predictor}'
                directory = output / 'bundles' / name
                if not (directory / 'codec.json').exists():
                    pack_codec_native(direct, directory, native_library=args.accelerated_library,
                        max_block_bytes=None, block_rows=block_rows, predictor=predictor)
                files = serving_files(directory)
                built = {**old_report, 'name': name, 'row_bundle_bytes': sum(f['bytes'] for f in files.values()),
                    'serving_files': files, 'layout_source_directory': str(direct),
                    'layout_source_metadata_sha256': digest(direct / 'codec.json')}
                built['model_bytes'] = built['row_bundle_bytes'] + built['common_model_bytes']
                report = output / 'reports' / (name + '.json')
                write(report, built)
                for variant, library in (('prior', args.prior_library), ('zlib', args.zlib_library), ('libdeflate', args.accelerated_library)):
                    configured = {**job, 'backend': 'faiss', 'directory': str(directory), 'name': name,
                        'encoding': codec, 'report': str(report), 'native_library': str(library)}
                    path = output / 'jobs' / (name + '-' + variant + '.json')
                    write(path, configured)
                    entries.append({'codec': codec, 'variant': variant, 'block_rows': block_rows,
                                    'predictor': predictor, 'job': str(path)})
    write(output / 'prepared.json', {'status': 'complete', 'source': str(source), 'entries': entries,
        'source_sha256': digest(__file__), 'libraries': {str(p): digest(p) for p in
        (args.prior_library, args.zlib_library, args.accelerated_library)}})
    return str(output / 'prepared.json')


def prepare(args):
    sources = sorted(p for p in args.sources.iterdir() if p.is_dir() and (p / 'jobs.json').exists())
    if len(sources) != 6:
        raise ValueError('all six trained Qwen sources required')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(lambda p: prepare_model(args, p), sources))
    write(args.output / 'prepared.json', {'status': 'complete', 'models': completed})


def lane(args):
    """Fixed GPU/core assignment; sequential methods with rotating order.

    Selection is CPU calibration timing only. Official GPU test timing audits
    that frozen choice; all unselected CPU cells remain in the grid.
    """
    import numpy as np
    cfg = load(args.prepared); root = args.output
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    entries = cfg['entries']
    from .benchmark_llm_codec_serving import requests_for
    from .codec_memory_adapter import CodecMultiHeadEmbedding
    from .late_memory_replay import load_engram
    from .ngram_identity import Vocabulary, IdentityMapping
    from .tree_node_quality import Reader, environment, memory, array_digest
    import faiss
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1); torch.set_num_interop_threads(1); faiss.omp_set_num_threads(1)
    first_job = load(entries[0]['job']); common = Path(first_job['common'])
    metadata = load(common / 'metadata.json'); recipe = load(common / 'model-recipe.json')
    module = load_engram(args.upstream); p = recipe['protocol']
    tokenizer = AutoTokenizer.from_pretrained(recipe['base_model'], local_files_only=True)
    config = load(Path(recipe['base_model']) / 'config.json')
    layer = config.get('text_config', config)['num_hidden_layers'] - p['suffix_layers']
    source_mapping = module.NgramHashMapping(engram_vocab_size=p['variants'][recipe['variant']]['vocab_sizes'],
        max_ngram_size=4, n_embed_per_ngram=64, n_head_per_ngram=2, layer_ids=[layer],
        tokenizer_name_or_path=recipe['base_model'], pad_id=tokenizer.pad_token_id, seed=recipe['seed'])
    mapping = IdentityMapping(source_mapping, Vocabulary.load(common / 'vocabulary.npz'), True)
    inputs = {split: [np.asarray([r['ids']], dtype=np.int64) for r in requests_for(common, split, args.requests)]
              for split in ('calibration', 'test')}
    traces = {split: [mapping.hash(x)[layer] for x in values] for split, values in inputs.items()}
    completed = []; reference_hashes = {}
    for entry in entries:
        name = Path(entry['job']).stem; out = root / 'cpu' / (name + '.json')
        if not out.exists() or load(out).get('status') != 'passed':
            out.parent.mkdir(parents=True, exist_ok=True)
            job = load(entry['job'])
            staged = CodecMultiHeadEmbedding(Reader(job), metadata['head_sizes'], 'cpu', zero_sentinels=True)
            report = {'status': 'running', 'environment': environment(), 'source_sha256': digest(__file__),
                'job_sha256': digest(entry['job']), 'query': {}, 'row_only': {},
                'cache_bytes': 0, 'rss_before': memory(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'contract': 'Real original-token queries, same warmed canonical-address mapping and unchanged code bytes. New row handles; warm OS cache, not disk-cold.',
                'decoded_row_sha256': load(job['report'])['decoded_row_sha256'],
                'complete_branch_bytes': load(job['report'])['model_bytes']}
            try:
                for split in inputs:
                    first = []; hashes = []
                    for x in inputs[split]:
                        start = time.perf_counter_ns(); result = staged(torch.from_numpy(mapping.hash(x)[layer]))
                        first.append((time.perf_counter_ns() - start) / 1e6); hashes.append(array_digest(result.numpy()))
                    if entry['variant'] == 'direct':
                        reference_hashes[(entry['codec'], split)] = hashes
                    elif hashes != reference_hashes[(entry['codec'], split)]:
                        raise ValueError('native layout/reader changed actual query vectors')
                    for _ in range(2):
                        for local in traces[split]: staged(torch.from_numpy(local))
                    full_samples = []; row_samples = []
                    for _ in range(5):
                        for x, local in zip(inputs[split], traces[split]):
                            start = time.perf_counter_ns(); staged(torch.from_numpy(mapping.hash(x)[layer]))
                            full_samples.append((time.perf_counter_ns() - start) / 1e6)
                            start = time.perf_counter_ns(); staged(torch.from_numpy(local))
                            row_samples.append((time.perf_counter_ns() - start) / 1e6)
                    for name, samples in (('query', full_samples), ('row_only', row_samples)):
                        report[name][split] = {'samples_ms': samples, 'p50_ms': float(np.median(samples)),
                            'p95_ms': float(np.quantile(samples, .95)), 'p99_ms': float(np.quantile(samples, .99))}
                    report['query'][split].update(first_ms=first, output_hashes=hashes,
                        input_hashes=[array_digest(x) for x in inputs[split]])
                native = staged.reader.reader.native
                report.update(status='passed', rss_after=memory(), staging_stats=staged.stats,
                    decoder_name=None if native is None else native.decoder_name)
            except BaseException as error:
                report.update(status='failed', error=repr(error)); raise
            finally:
                staged.close(); write(out, report)
        else:
            previous = load(out)
            if entry['variant'] == 'direct':
                for split in inputs: reference_hashes[(entry['codec'], split)] = previous['query'][split]['output_hashes']
        completed.append({**entry, 'report': str(out)})
        write(root / 'cpu-progress.json', {'status': 'running', 'completed': completed, 'expected': len(entries)})
    selected = []
    for codec in sorted({e['codec'] for e in entries}):
        candidates = [e for e in completed if e['codec'] == codec and e['variant'] == 'libdeflate']
        # Official calibration prefixes determine the layout, never test times.
        chosen = min(candidates, key=lambda e: (load(e['report'])['query']['calibration']['p50_ms'], Path(e['job']).stem))
        selected.append(chosen)
    selection = {'rule': 'Lowest complete-token CPU calibration p50 within each unchanged decoded codec; test never reselects.',
                 'selected': selected, 'cpu_reports_sha256': {e['report']: digest(e['report']) for e in completed}}
    write(root / 'selection.json', selection)
    write(root / 'cpu-progress.json', {'status': 'complete', 'completed': completed, 'expected': len(entries)})
    if args.cpu_only:
        return
    for codec, chosen in [(e['codec'], e) for e in selected]:
        reference = [e for e in entries if e['codec'] == codec and e['variant'] == 'direct'][0]
        matching = [e for e in entries if e['codec'] == codec and e.get('block_rows') == chosen['block_rows']
                    and e.get('predictor') == chosen['predictor']]
        variants = [reference, *matching]
        for cohort in range(3):
            for entry in variants[cohort:] + variants[:cohort]:
                name = Path(entry['job']).stem
                for batch_size in (1, 4):
                    out = root / 'gpu' / f'{codec}-cohort{cohort}-batch{batch_size}-{name}.json'
                    if out.exists() and load(out).get('status') == 'passed':
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-m', 'tools.benchmark_llm_codec_serving', 'gpu',
                        '--job', entry['job'], '--output', str(out), '--upstream', str(args.upstream),
                        '--requests', '8', '--repeat', '3', '--warmup', '2', '--decode-steps', '16',
                        '--batch-size', str(batch_size), '--skip-spare-routing']
                    with out.with_suffix('.log').open('w') as log:
                        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root / 'complete.json', {'status': 'complete', 'selection_sha256': digest(root / 'selection.json'),
        'source_sha256': digest(__file__), 'cuda_selector': env.get('CUDA_VISIBLE_DEVICES'),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'gpu_reports': {str(p): digest(p) for p in sorted((root / 'gpu').glob('*.json'))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare')
    for name in ('sources', 'output', 'prior-library', 'zlib-library', 'accelerated-library'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--codecs', default='fp32,sq4,sq6,sq8,pq-8x8,pq-16x8')
    p.add_argument('--workers', type=int, default=6)
    p = sub.add_parser('lane')
    for name in ('prepared', 'output', 'upstream'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--requests', type=int, default=32)
    p.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare': prepare(args)
    else: lane(args)


if __name__ == '__main__':
    main()
