"""Analyze existing trained n-gram vectors without training or GPU access."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

import numpy as np

from .conditional_similarity import audit_relation, structural_pairs, matched_controls
from .native_row_store import NativeRows, pack_native

ROOT = Path(__file__).resolve().parents[1]
CACHE = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def metadata(text, order, settings, seed):
    from .train_ngram_similarity import ngrams
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    counts = Counter(tokens)
    words = sorted((w for w in counts if counts[w] >= settings['min_count']),
                   key=lambda w: (-counts[w], w))[:settings['vocab_limit']]
    lookup = {w: i for i, w in enumerate(words)}
    bags = [Counter(ngrams(w, order)) for w in words]
    keys = sorted({g for bag in bags for g in bag})
    signatures = defaultdict(list)
    for i, bag in enumerate(bags):
        for g, multiplicity in bag.items():
            signatures[g].append((i, multiplicity))
    seq = [lookup.get(w, -1) for w in tokens]
    # Recreate the exact context-pool center order, preserving OOV positions.
    centers = np.asarray([seq[i] for i in range(len(seq)) if seq[i] >= 0
        for j in range(max(0, i - 2), min(len(seq), i + 3)) if i != j and seq[j] >= 0], dtype=np.int64)
    events = np.zeros(len(words), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(settings['steps']):
        np.add.at(events, centers[rng.integers(len(centers), size=settings['batch_size'])], 1)
    counts_array = np.array([sum(counts[words[i]] * m for i, m in signatures[g]) for g in keys], dtype=np.int64)
    exposures = np.array([sum(events[i] * m for i, m in signatures[g]) for g in keys], dtype=np.int64)
    unique_signatures = {s: i for i, s in enumerate(sorted({tuple(signatures[g]) for g in keys}))}
    signature_ids = np.array([unique_signatures[tuple(signatures[g])] for g in keys], dtype=np.int64)
    return words, keys, counts_array, exposures, signature_ids, events


def comparisons(keys, counts, relation):
    h = hashlib.sha256(counts.astype('<i8').tobytes())
    h.update(json.dumps(keys, separators=(',', ':')).encode())
    cache_key = (h.hexdigest(), relation)
    if cache_key not in CACHE:
        pairs, population = structural_pairs(keys, relation)
        controls = matched_controls(keys, counts, pairs)
        child_controls = matched_controls(keys, counts, pairs[:, ::-1]) if relation in ('parent', 'ancestor') else None
        CACHE[cache_key] = (pairs, population, controls, child_controls)
    return CACHE[cache_key]


def analyze(rows, keys, counts, relations, **kwargs):
    return [audit_relation(rows, keys, counts, r, prepared=comparisons(keys, counts, r), **kwargs) for r in relations]


def storage(rows, keys, counts, out, library):
    from .benchmark_cosine_merge import structural_merge, fidelity
    source = out / 'source.f32'
    source.write_bytes(np.asarray(rows, dtype='<f4').tobytes())
    baseline = pack_native(source, out / 'identity.men', rows.shape[1] * 4, 64, False, 'xor')
    results = []
    for ratio in (.99, .95):
        mapping = structural_merge(rows, keys, .99, ratio)
        table = out / f'compact-r{ratio}.men'
        packed = pack_native(source, table, rows.shape[1] * 4, 64, False, 'xor', representatives=mapping.tolist())
        approximated = rows[mapping]
        with NativeRows(table, library) as native:
            h = hashlib.sha256()
            for start in range(0, len(rows), 4096):
                h.update(native.rows(range(start, min(start + 4096, len(rows)))))
        if h.hexdigest() != hashlib.sha256(approximated.astype('<f4').tobytes()).hexdigest():
            raise AssertionError('native reconstructed approximation differs')
        changed = mapping != np.arange(len(rows))
        results.append({'minimum_cosine': .99, 'minimum_norm_ratio': ratio,
            'merged_rows': int(changed.sum()), 'retained_rows': int((~changed).sum()),
            'changed_occurrence_mass': int(counts[changed].sum()), 'total_occurrence_mass': int(counts.sum()),
            'packed': packed, 'fidelity': fidelity(rows, approximated, counts),
            'approximation_sha256': h.hexdigest(), 'native_reads_approximation_byte_exact': True})
    return {'scope': 'N-gram rows and native map/codec only; excludes keys/word/output weights and task quality',
            'identity': baseline, 'approximate': results}


def legacy(args, report):
    training = json.loads((args.training_root / 'report.json').read_text())
    if digest(args.corpus) != training['corpus_sha256']:
        raise ValueError('legacy source corpus changed')
    text = args.corpus.read_text()
    for r in training['runs']:
        order, seed = r['max_order'], r['seed']
        words, keys, counts, exposure, signatures, _ = metadata(text, order, training['settings'], seed)
        path = args.training_root / f'order-{order}-seed-{seed}' / 'trained-ngram.f32'
        if digest(path) != r['learned_row_sha256'] or len(keys) != r['ngram_rows']:
            raise ValueError('legacy trained row source changed')
        rows = np.memmap(path, mode='r', dtype='<f4', shape=(len(keys), training['settings']['dim']))
        out = args.output / f'legacy-order-{order}-s{seed}'
        out.mkdir()
        result = {'dataset': 'tinyshakespeare', 'convention': 'legacy', 'order': order, 'seed': seed,
            'source_row_sha256': digest(path), 'source_training_report_sha256': digest(args.training_root / 'report.json'),
            'corpus_sha256': digest(args.corpus), 'rows': len(keys), 'dimension': rows.shape[1],
            'initialization': 'Not retained by original run; no initialization reconstruction claimed',
            'exposures': 'Recreated from exact saved sampling algorithm, seed and source context pool; no original center-event hash retained',
            'relations': analyze(rows, keys, counts, ('parent', 'ancestor'), signatures=signatures, exposures=exposure),
            'storage': storage(rows, keys, counts, out, args.native_library)}
        publish(args, report, out.name, result)


def checkpoints(args, report):
    import torch
    training = json.loads((args.training_root / 'report.json').read_text())
    corpus = args.training_root / 'corpus.txt'
    if training['status'] != 'completed' or digest(corpus) != training['tokenized_corpus_sha256']:
        raise ValueError('complete verified checkpoint corpus required')
    from .benchmark_heldout_compression import split_tokens
    text = ' '.join(split_tokens(corpus.read_text())['train'])
    for r in training['runs']:
        order, seed, convention = r['order'], r['seed'], r['convention']
        path = args.training_root / convention / f'order-{order}-seed-{seed}' / 'reference.pt'
        if digest(path) != r['checkpoint_sha256']:
            raise ValueError('trained checkpoint changed')
        c = torch.load(path, map_location='cpu', weights_only=True)
        words, keys, counts, exposure, signatures, events = metadata(text, order, training['settings'], seed)
        event_hash = hashlib.sha256(events.astype('<u8').tobytes()).hexdigest()
        if words != c['words'] or keys != c['grams'] or event_hash != r['sampled_center_events_sha256']:
            raise ValueError('training-only registry or actual sampling exposure changed')
        rows = c['input_rows'][1 + len(words):].numpy()
        initial = c['initial_rows'][1 + len(words):].numpy()
        out = args.output / f'{convention}-order-{order}-s{seed}'
        out.mkdir()
        result = {'dataset': 'enwik8', 'convention': convention, 'order': order, 'seed': seed,
            'source_checkpoint_sha256': digest(path), 'source_training_report_sha256': digest(args.training_root / 'report.json'),
            'sampled_center_events_sha256': event_hash, 'corpus_sha256': digest(corpus),
            'rows': len(rows), 'dimension': rows.shape[1],
            'relations': analyze(rows, keys, counts, ('parent', 'ancestor'), initial=initial, signatures=signatures, exposures=exposure),
            'storage': storage(rows, keys, counts, out, args.native_library)}
        publish(args, report, out.name, result)


def qwen(args, report):
    import torch
    from .late_memory_replay import load_engram
    from .ngram_identity import Vocabulary, reference_mapping
    for fixture in sorted(args.training_root.iterdir()):
        mpath = fixture / 'manifest.json'
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        if m['status'] != 'passed' or digest(fixture / 'rows.f32') != m['source_row_sha256']:
            raise ValueError('original trained Qwen rows changed')
        vocabulary = Vocabulary.load(fixture / 'vocabulary.npz')
        if digest(fixture / 'vocabulary.npz') != m['source_vocabulary_sha256']:
            raise ValueError('canonical vocabulary changed')
        recipe = json.loads((fixture / 'xor/model-recipe.json').read_text())
        base = Path(recipe['base_model'])
        config = json.loads((base / 'config.json').read_text()); config = config.get('text_config', config)
        layer = config['num_hidden_layers'] - recipe['protocol']['suffix_layers']
        module = load_engram(args.upstream)
        source = module.NgramHashMapping(engram_vocab_size=recipe['protocol']['variants'][recipe['variant']]['vocab_sizes'],
            max_ngram_size=4, n_embed_per_ngram=64, n_head_per_ngram=2, layer_ids=[layer],
            tokenizer_name_or_path=str(base), pad_id=vocabulary.metadata['pad_id'], seed=recipe['seed'])
        source = reference_mapping(source, vocabulary)
        rows = np.memmap(fixture / 'rows.f32', mode='r', dtype='<f4', shape=tuple(m['shape']))
        out = args.output / fixture.name; out.mkdir()
        heads = []
        for head, (offset, size) in enumerate(zip(m['offsets'], m['head_sizes'])):
            order = 2 + head // 2
            keys = [tuple(row) for row in vocabulary.keys[order].tolist()]
            counts = vocabulary.counts[order]
            original_hashes = source._get_ngram_hashes(vocabulary.keys[order], layer)[:, -1, head]
            generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(
                f"{recipe['seed']}:identity:{order}:{head % 2}".encode()).digest()[:8], 'little'))
            reference = torch.empty(size, 32)
            torch.nn.init.normal_(reference, mean=0., std=config['initializer_range'], generator=generator)
            reference[0].zero_()
            initial = reference[torch.from_numpy(original_hashes + 1)].numpy()
            learned = rows[offset + 1:offset + 1 + len(keys)]
            result = {'head': head, 'order': order, 'rows': len(keys),
                'initialization': 'Exact original per-head seed, normal scale and matched hash-bucket cloning',
                'relations': analyze(learned, keys, counts, ('prefix', 'suffix'), initial=initial)}
            heads.append(result)
        publish(args, report, out.name, {'dataset': 'multicounterfact', 'model': fixture.name.split('-')[0],
            'seed': recipe['seed'], 'source_fixture_manifest_sha256': digest(mpath),
            'source_row_sha256': m['source_row_sha256'], 'vocabulary_sha256': digest(fixture / 'vocabulary.npz'),
            'heads': heads, 'scope': 'Within-order/head canonical token siblings; no cross-head parent cosine or independent length effect'})


def publish(args, report, name, result):
    write(args.output / name / 'analysis.json', result)
    report['runs'].append({'name': name, 'analysis_sha256': digest(args.output / name / 'analysis.json')})
    write(args.output / 'report.json', report)
    print(json.dumps({'completed': name, 'sources': len(report['runs'])}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=('legacy', 'checkpoints', 'qwen'))
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus', type=Path)
    p.add_argument('--upstream', type=Path)
    p.add_argument('--native-library', type=Path)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.output.exists():
        p.error('CPU-only requires CUDA_VISIBLE_DEVICES empty and fresh output')
    if args.kind == 'legacy' and (not args.corpus or not args.native_library):
        p.error('legacy requires corpus and native library')
    if args.kind == 'checkpoints' and not args.native_library:
        p.error('checkpoints require native library')
    if args.kind == 'qwen' and not args.upstream:
        p.error('qwen requires pinned upstream source')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    cpu = platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():
        cpu = next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
    report = {'status': 'running', 'kind': args.kind, 'runs': [], 'environment': {
        'command': shlex.join(sys.orig_argv), 'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
        'os': platform.platform(), 'cpu': cpu, 'gpu': 'unused', 'threads': 1, 'interop_threads': 1,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': digest(__file__), 'analysis_source_sha256': digest(ROOT / 'tools/conditional_similarity.py')},
        'protocol_sha256': digest(ROOT / 'docs/conditional_similarity_protocol_20261109.md'),
        'cost_assumptions': 'Existing CPU and existing trained vectors; no training, GPU work, downloads or paid services',
        'limitations': ['Post-hoc existing-checkpoint analysis; known aggregate results before fixed new strata.',
            'Finite small SGNS and frozen-backbone editing, not production/pretraining or universal similarity.',
            'Sibling sampling is uniform over pairs; observed unique-row coverage is not extrapolated.',
            'Matched controls use count bins; disclose residual imbalance and unmatched pairs.',
            'Spearman coefficients describe correlated pair observations; no independent-pair significance claim.',
            'Approximate row sharing is not source-byte losslessness or a task-quality guarantee.']}
    {'legacy': legacy, 'checkpoints': checkpoints, 'qwen': qwen}[args.kind](args, report)
    report['status'] = 'complete'; write(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
