"""Real trained identity checkpoints: lossless storage and CPU row-query costs.

No backbone is loaded or trained. Original physical rows, including sentinels
and budget padding, are retained. Model quality/whole-model latency are separate
measurements; source editing scores are provenance, not a native quality check.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from .benchmark_device_gather import rss
from .engram_row_adapter import MappedMultiHeadEmbedding, export_multihead, file_sha256, head_offsets
from .late_memory_replay import load_engram
from .ngram_identity import IdentityMapping, Vocabulary
from .structural_native import StructuralPlan, StructuralIndex

ROOT = Path(__file__).resolve().parents[1]
ROW_NAME = 'ngram_memory.multi_head_embedding.embedding.weight'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def environment():
    cpu = platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    return {'python': sys.version, 'pytorch': torch.__version__, 'numpy': np.__version__,
            'os': platform.platform(), 'cpu': cpu, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'device': 'cpu', 'gpu': 'unused',
            'command': shlex.join([sys.executable, *sys.argv]),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': file_sha256(__file__)}


def canonical_key(head, tokens):
    # Fixed-width encoding retains chronological token identity and shared
    # prefixes/suffixes. Tags at both ends separate the vector heads.
    return f'h{head}/' + ''.join(f'{int(t) & ((1 << 64) - 1):016x}' for t in tokens) + f'/h{head}'


def prepare(args):
    run = args.run
    r = json.loads((run / 'report.json').read_text())
    recipe = json.loads((run / 'model-recipe.json').read_text())
    if r['status'] != 'complete' or r['identity']['addressing'] != 'exact-canonical-key':
        raise ValueError('completed exact-canonical-key checkpoint required')
    for field in ('protocol', 'seed', 'variant', 'identity'):
        if recipe[field] != r[field]:
            raise ValueError('source recipe/report mismatch: ' + field)
    delta = run / 'delta.safetensors'
    if file_sha256(delta) != r['delta']['sha256'] or recipe['delta_sha256'] != r['delta']['sha256']:
        raise ValueError('source delta checksum changed')
    if file_sha256(run / 'vocabulary.npz') != r['identity']['vocabulary_sha256']:
        raise ValueError('source vocabulary checksum changed')
    prepared = json.loads((args.prepared / 'prepare.json').read_text())
    if (file_sha256(args.prepared / 'prepare.json') != r['prepared_sha256']
            or file_sha256(args.prepared / 'sequences.json') != prepared['payload_sha256']):
        raise ValueError('prepared evaluation requests changed')
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    with safe_open(delta, framework='pt', device='cpu') as state:
        if set(state.keys()) != set(r['trained_parameter_names']):
            raise ValueError('registered source delta parameters changed')
        weight = state.get_tensor(ROW_NAME)
        nonrow = {name: state.get_tensor(name) for name in state.keys() if name != ROW_NAME}
    heads = r['identity']['head_sizes']
    if weight.dtype != torch.float32 or list(weight.shape) != [sum(heads), 32]:
        raise ValueError('source head shape/dtype changed')
    raw = weight.numpy().astype('<f4', copy=False).tobytes()
    if len(raw) != r['identity']['row_bytes']:
        raise ValueError('source row-byte budget changed')
    (out / 'rows.f32').write_bytes(raw)
    save_file(nonrow, str(out / 'nonrow-delta.safetensors'))
    shutil.copyfile(run / 'vocabulary.npz', out / 'vocabulary.npz')
    del nonrow
    cfg = json.loads((args.base / 'config.json').read_text())
    text_cfg = cfg.get('text_config', cfg)
    layer = text_cfg['num_hidden_layers'] - r['protocol']['suffix_layers']
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    module = load_engram(args.upstream)
    source = module.NgramHashMapping(engram_vocab_size=r['protocol']['variants'][r['variant']]['vocab_sizes'],
        max_ngram_size=4, n_embed_per_ngram=64, n_head_per_ngram=2, layer_ids=[layer],
        tokenizer_name_or_path=str(args.base), pad_id=tokenizer.pad_token_id, seed=r['seed'])
    vocab = Vocabulary.load(out / 'vocabulary.npz')
    mapping = IdentityMapping(source, vocab, True)
    configured_heads = [n for order in mapping.vocab_size_across_layers[layer] for n in order]
    if configured_heads != heads:
        raise ValueError('training identity addressing differs from stored heads')
    payload = json.loads((args.prepared / 'sequences.json').read_text())
    queries, cases = {}, []
    offsets = np.asarray(head_offsets(heads), dtype=np.int64)
    for c in payload['cases']:
        if c['split'] != 'test':
            continue
        for group in ('rewrite', 'paraphrase', 'neighborhood'):
            for pair in c['groups'][group]:
                seq = payload['sequences'][pair['new']]
                ids = np.asarray([seq['ids'][:seq['target_start']]], dtype=np.int64)
                local = mapping.hash(ids)[layer]
                key = 'q' + str(len(queries))
                queries[key] = local
                expected = weight.numpy()[local + offsets]
                cases.append({'key': key, 'case_id': c['case_id'], 'group': group,
                              'tokens': ids.shape[1], 'input_ids_sha256': sha(ids.astype('<i8').tobytes()),
                              'query_ids_sha256': sha(local.astype('<i8').tobytes()),
                              'output_sha256': sha(expected.tobytes())})
                if len(cases) == args.requests:
                    break
            if len(cases) == args.requests:
                break
        if len(cases) == args.requests:
            break
    if len(cases) != args.requests:
        raise ValueError('insufficient official test prefixes')
    np.savez(out / 'queries.npz', **queries)
    del payload, mapping, source, tokenizer
    gc.collect()
    base_bytes = sum(p.stat().st_size for p in args.base.rglob('*') if p.is_file())
    bundle_recipe = {'base_model': str(args.base), 'seed': r['seed'], 'variant': r['variant'],
        'protocol': r['protocol'], 'global_step': r['optimizer_steps'], 'nonrow_delta': 'nonrow-delta.safetensors',
        'nonrow_delta_sha256': file_sha256(out / 'nonrow-delta.safetensors'),
        'memory': 'memory/embedding.json', 'vocabulary': 'vocabulary.npz',
        'vocabulary_sha256': r['identity']['vocabulary_sha256'], 'identity': r['identity'],
        'prepared_sha256': r['prepared_sha256'], 'native_oov': False,
        'frozen_weight_sha256': r['frozen_weight_sha256'],
        'frozen_decoder_suffix_sha256': r['frozen_decoder_suffix_sha256'],
        'contract': 'Original trained identity rows and nonrow delta; original immutable base required'}
    entry = {'status': 'building', 'environment': environment(), 'run_name': run.name,
        'source_run_report_sha256': file_sha256(run / 'report.json'),
        'source_recipe_sha256': file_sha256(run / 'model-recipe.json'),
        'source_delta_sha256': r['delta']['sha256'], 'source_delta_bytes': r['delta']['bytes'],
        'source_identity_bundle_bytes': sum((run / name).stat().st_size for name in
                                            ('delta.safetensors', 'vocabulary.npz', 'model-recipe.json')),
        'source_vocabulary_sha256': r['identity']['vocabulary_sha256'],
        'prepared_sha256': r['prepared_sha256'], 'source_row_sha256': sha(raw),
        'shape': list(weight.shape), 'head_sizes': heads, 'offsets': offsets.tolist(),
        'row_bytes': len(raw), 'logical_exact_rows_with_zero_sentinels': r['identity']['logical_exact_rows_with_zero_sentinels'],
        'unaddressed_budget_padding_rows': sum(heads) - r['identity']['logical_exact_rows_with_zero_sentinels'],
        'vocabulary_bytes': (out / 'vocabulary.npz').stat().st_size,
        'nonrow_bytes': (out / 'nonrow-delta.safetensors').stat().st_size,
        'shared_base_directory_bytes': base_bytes, 'source_editing_metrics': r['metrics'],
        'queries_sha256': file_sha256(out / 'queries.npz'), 'cases': cases, 'variants': [],
        'cost_assumptions': 'Existing shared CPU; no new training, paid API or rented compute. Monetary/energy cost unmeasured.',
        'limitations': ['Row storage and CPU gather only; no native whole-model quality or latency claim.',
            'Physical training-budget padding retained and disclosed; no capacity extrapolation to billion-row tables.',
            'DATrie uses reversible encoded canonical token tuples, not decoded surface-text n-grams.',
            'Original Python identity dictionary remains required by the Qwen integration; optional DATrie is an added index here.',
            'No approximation, quantization, full-model pretraining, generated-answer or cold-disk evaluation.']}
    started = time.perf_counter()
    with (out / 'rows.f32.gz').open('wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0, compresslevel=6) as z:
            z.write(raw)
    if gzip.decompress((out / 'rows.f32.gz').read_bytes()) != raw:
        raise AssertionError('gzip roundtrip differs')
    entry['gzip'] = {'bytes': (out / 'rows.f32.gz').stat().st_size, 'build_and_verify_seconds': time.perf_counter() - started,
                     'level': 6, 'all_original_bytes_verified': True, 'random_access': False}
    policies = [('identity', 'none', False), ('xor', 'xor', False), ('dedup-xor', 'xor', True)]
    for label, predictor, dedup in policies:
        directory = out / label
        directory.mkdir()
        started = time.perf_counter()
        export_multihead(weight, heads, directory / 'memory', block_rows=args.block_rows,
                         backend='native', deduplicate=dedup, predictor=predictor)
        publish_variant(directory, bundle_recipe, entry, started, args)
    if args.structural_library:
        keys = {}
        for head, offset in enumerate(offsets):
            order = 2 + head // 2
            for i, tokens in enumerate(vocab.keys[order]):
                keys[int(offset) + i + 1] = canonical_key(head, tokens)
        for mode in ('layout', 'exact'):
            directory = out / ('datrie-' + mode)
            directory.mkdir()
            started = time.perf_counter()
            with StructuralPlan(raw, 32, keys, args.structural_library, mode=mode) as plan:
                plan.export(out / 'rows.f32', directory / 'memory', block_rows=args.block_rows, predictor='xor')
                if not plan.original_bytes_preserved:
                    raise AssertionError('lossless structural planner changed source rows')
                merged = sum(i != r for i, r in enumerate(plan.representatives))
                meta = json.loads((directory / 'memory/embedding.json').read_text())
                meta.update(head_sizes=heads, offsets=offsets.tolist())
                write(directory / 'memory/embedding.json', meta)
                # Verify all registered keys through the persisted index, not
                # only the builder's in-memory map.
                with StructuralIndex(directory / 'memory/keys.das', args.structural_library) as index:
                    keyed = list(keys)
                    for start in range(0, len(keyed), 4096):
                        ids = keyed[start:start+4096]
                        if index.lookup([keys[i] for i in ids]) != ids:
                            raise AssertionError('persisted DATrie changed original IDs')
            publish_variant(directory, bundle_recipe, entry, started, args, merged=merged)
    entry.update(status='passed', native_library_sha256=file_sha256(args.native_library),
                 structural_library_sha256=file_sha256(args.structural_library) if args.structural_library else None)
    write(out / 'manifest.json', entry)
    print(json.dumps({'run': run.name, 'status': 'passed', 'row_bytes': len(raw),
                      'variants': [{k: v[k] for k in ('name', 'memory_artifact_bytes', 'bundle_bytes', 'structurally_merged_rows')} for v in entry['variants']]}), flush=True)


def publish_variant(directory, recipe, entry, started, args, *, merged=0):
    memory = directory / 'memory'
    with MappedMultiHeadEmbedding(memory, native_library=args.native_library) as mapped:
        digest = hashlib.sha256()
        for start in range(0, entry['shape'][0], 4096):
            digest.update(mapped.table.rows(range(start, min(start + 4096, entry['shape'][0]))))
        if digest.hexdigest() != entry['source_row_sha256']:
            raise AssertionError('original-ID row roundtrip differs')
    for name in ('nonrow-delta.safetensors', 'vocabulary.npz'):
        shutil.copyfile(args.output / name, directory / name)
    write(directory / 'model-recipe.json', {**recipe, 'memory_manifest_sha256': file_sha256(memory / 'embedding.json')})
    memory_bytes = sum(p.stat().st_size for p in memory.iterdir() if p.is_file())
    bundle_bytes = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    entry['variants'].append({'name': directory.name, 'memory_artifact_bytes': memory_bytes,
        'bundle_bytes': bundle_bytes, 'full_bundle_with_shared_base_bytes': bundle_bytes + entry['shared_base_directory_bytes'],
        'source_bundle_to_native_bundle_factor': entry['source_identity_bundle_bytes'] / bundle_bytes,
        'row_payload_to_memory_artifact_factor': entry['row_bytes'] / memory_bytes,
        'build_and_verify_seconds': time.perf_counter() - started,
        'table_sha256': file_sha256(memory / 'table.men'),
        'memory_manifest_sha256': file_sha256(memory / 'embedding.json'),
        'recipe_sha256': file_sha256(directory / 'model-recipe.json'),
        'all_original_rows_byte_exact': True, 'structurally_merged_rows': merged,
        'datrie_index_bytes': (memory / 'keys.das').stat().st_size if (memory / 'keys.das').exists() else 0})
    print(json.dumps({'run': entry['run_name'], 'completed_variant': directory.name,
                      'memory_bytes': memory_bytes, 'merged': merged}), flush=True)


def worker(args):
    manifest = json.loads((args.fixture / 'manifest.json').read_text())
    if manifest['status'] != 'passed' or file_sha256(args.fixture / 'queries.npz') != manifest['queries_sha256']:
        raise ValueError('verified fixture and original requests required')
    with np.load(args.fixture / 'queries.npz', allow_pickle=False) as f:
        queries = [torch.from_numpy(f[c['key']].copy()) for c in manifest['cases']]
    for c, q in zip(manifest['cases'], queries):
        if sha(q.numpy().astype('<i8').tobytes()) != c['query_ids_sha256']:
            raise ValueError('query checksum changed')
    report = {'status': 'running', 'mode': args.mode, 'environment': environment(),
              'manifest_sha256': file_sha256(args.fixture / 'manifest.json'), 'warmup': args.warmup,
              'repeat': args.repeat, 'rss_before_open': rss(), 'samples_ms': [], 'cpu_samples_ms': []}
    started = time.perf_counter()
    offsets = torch.tensor(manifest['offsets'], dtype=torch.int64)
    limits = torch.tensor(manifest['head_sizes'], dtype=torch.int64)
    def global_ids(ids):
        if bool(((ids < 0) | (ids >= limits)).any()):
            raise IndexError('head-local ID out of range')
        return (ids + offsets).contiguous().reshape(-1)
    mapped = None
    if args.mode in ('dense', 'raw-mmap'):
        if file_sha256(args.fixture / 'rows.f32') != manifest['source_row_sha256']:
            raise ValueError('raw source checksum changed')
        array = np.memmap(args.fixture / 'rows.f32', mode='r', dtype='<f4', shape=tuple(manifest['shape']))
        if args.mode == 'dense':
            weight = torch.from_numpy(np.array(array, copy=True))
            del array
            def forward(ids):
                return weight.index_select(0, global_ids(ids)).reshape(*ids.shape, 32)
        else:
            def forward(ids):
                return torch.from_numpy(array[global_ids(ids).numpy()]).reshape(*ids.shape, 32)
    else:
        variant = next(v for v in manifest['variants'] if v['name'] == args.mode)
        memory = args.fixture / args.mode / 'memory'
        if (file_sha256(memory / 'table.men') != variant['table_sha256']
                or file_sha256(memory / 'embedding.json') != variant['memory_manifest_sha256']):
            raise ValueError('native fixture changed')
        mapped = MappedMultiHeadEmbedding(memory, native_library=args.native_library,
            native_cache_bytes=args.cache_bytes, native_cache_policy='rows')
        forward = mapped
    gc.collect()
    report.update(open_seconds=time.perf_counter() - started, rss_after_open=rss(),
                  cache_bytes=args.cache_bytes if mapped else 0)
    try:
        with torch.no_grad():
            first = []
            for c, ids in zip(manifest['cases'], queries):
                started = time.perf_counter_ns()
                output = forward(ids)
                first.append((time.perf_counter_ns() - started) / 1e6)
                if sha(output.numpy().tobytes()) != c['output_sha256']:
                    raise AssertionError('query output differs from trained rows')
                del output
            report['first_trace_ms'] = first
            for _ in range(args.warmup):
                for ids in queries:
                    output = forward(ids)
                    del output
            for _ in range(args.repeat):
                for ids in queries:
                    cpu_start, started = time.process_time_ns(), time.perf_counter_ns()
                    output = forward(ids)
                    report['samples_ms'].append((time.perf_counter_ns() - started) / 1e6)
                    report['cpu_samples_ms'].append((time.process_time_ns() - cpu_start) / 1e6)
                    del output
        report.update(status='passed', all_query_outputs_byte_exact=True, rss_after_query=rss(),
            query_row_counts=[q.numel() for q in queries],
            p50_ms=float(np.median(report['samples_ms'])), p95_ms=float(np.percentile(report['samples_ms'], 95)),
            note='Real test-prefix row queries, fixed chronological replay, warm OS page cache. Excludes tokenizer/key lookup, GPU and model.')
        write(args.output, report)
    finally:
        if mapped:
            mapped.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for name in ('run', 'prepared', 'base', 'upstream', 'output', 'native-library'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--structural-library', type=Path)
    prep.add_argument('--block-rows', type=int, default=32)
    prep.add_argument('--requests', type=int, default=64)
    bench = sub.add_parser('worker')
    for name in ('fixture', 'native-library', 'output'):
        bench.add_argument('--' + name, type=Path, required=True)
    bench.add_argument('--mode', choices=('dense', 'raw-mmap', 'identity', 'xor', 'dedup-xor', 'datrie-layout', 'datrie-exact'), required=True)
    bench.add_argument('--cache-bytes', type=int, default=1048576)
    bench.add_argument('--warmup', type=int, default=3)
    bench.add_argument('--repeat', type=int, default=20)
    args = p.parse_args()
    if args.action == 'prepare' and (args.block_rows < 1 or not 1 <= args.requests <= 1024):
        p.error('positive block rows and 1..1024 real requests required')
    if args.action == 'worker' and (args.repeat < 1 or args.warmup < 0 or args.cache_bytes < 0):
        p.error('positive repeat, nonnegative warmup and cache required')
    if args.output.exists():
        p.error('fresh output required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    (prepare if args.action == 'prepare' else worker)(args)


if __name__ == '__main__':
    main()
