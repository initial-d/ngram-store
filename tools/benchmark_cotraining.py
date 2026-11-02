"""CPU-only co-training dynamics and certified aggregate-compaction evidence.

Actual SGNS checkpoints/updates, train-only incidence groups, held-out known-word
objectives and explicit OOV rejection. No arbitrary-row byte preservation or
LLM memory/convergence claim. The protocol is written before any run.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark_heldout_compression import (split_tokens, evaluation_set, compose,
                                           score, summarize, deltas)
from .cotraining_groups import BagPartition, CertifiedBagRows, grouped_embedding_bag
from .engram_row_adapter import file_sha256
from .native_row_store import NativeRows, pack_native
from .structural_native import StructuralPlan
from .train_ngram_similarity import ngrams, prepare


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'count': 0}
    return {'count': len(values), 'mean': float(values.mean()),
            'p10': float(np.quantile(values, .1)), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'minimum': float(values.min()),
            'maximum': float(values.max())}


def vector_metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    nonzero = (an > 0) & (bn > 0)
    cosine = np.zeros(len(a))
    np.divide((a * b).sum(1), an * bn, out=cosine, where=nonzero)
    ratio = np.zeros(len(a))
    np.divide(np.minimum(an, bn), np.maximum(an, bn), out=ratio, where=nonzero)
    error = np.linalg.norm(a - b, axis=1) / np.maximum(an, 1e-30)
    return {'cosine': distribution(cosine), 'norm_ratio': distribution(ratio),
            'joint099_fraction': float(np.mean(nonzero & (cosine >= .99) & (ratio >= .99))) if len(a) else None,
            'relative_l2': distribution(error),
            'global_relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))}


def fit_partition(bags, nrows, word_offset):
    registered = [[int(i) for i in bag if i != 0] for bag in bags.numpy()]
    return BagPartition.fit(registered, nrows, protected=range(word_offset))


def dynamics_snapshot(rows, initial, partition, grams, offset):
    rows, initial = rows.detach().numpy(), initial.detach().numpy()
    root = partition.representatives()
    selected = np.flatnonzero(root != np.arange(len(root)))
    a, b = rows[selected], rows[root[selected]]
    delta_a = a.astype(np.float64) - initial[selected]
    delta_b = b.astype(np.float64) - initial[root[selected]]
    result = {'identical_incidence_pairs': vector_metrics(a, b),
              'actual_update_pairs': vector_metrics(delta_a, delta_b),
              'update_relative_to_initial': distribution(
                  np.linalg.norm((rows - initial).astype(np.float64), axis=1)[offset:] /
                  np.maximum(np.linalg.norm(initial.astype(np.float64), axis=1)[offset:], 1e-30)),
              'maximum_initial_offset_drift': float(np.max(np.abs(delta_a - delta_b))) if len(a) else None}
    lookup = {g: i + offset for i, g in enumerate(grams)}
    pairs = sorted({(lookup[parent], child) for g, child in lookup.items()
                    for parent in (g[:-1], g[1:]) if parent in lookup})
    if pairs:
        index = np.asarray(pairs)
        result['structural_pairs'] = vector_metrics(rows[index[:, 0]], rows[index[:, 1]])
        same = partition.mapping[index[:, 0]] == partition.mapping[index[:, 1]]
        result['structural_identical_incidence_pairs'] = vector_metrics(rows[index[same, 0]], rows[index[same, 1]])
    return result


def train_dynamics(split, cfg, order, seed, convention, directory):
    words, grams, _, pairs, bags, lengths, counts = prepare(' '.join(split['train']), order,
                                                          cfg.vocab_limit, cfg.min_count)
    if len(words) < 2 or not len(pairs):
        raise ValueError('not enough registered training words/context pairs')
    offset, nrows = 1 + len(words), 1 + len(words) + len(grams)
    partition = fit_partition(bags, nrows, offset)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dense = nn.EmbeddingBag(nrows, cfg.dim, mode='sum', sparse=True, padding_idx=0)
    outputs = nn.Embedding(len(words), cfg.dim, sparse=True)
    with torch.no_grad():
        dense.weight.uniform_(-1 / cfg.dim, 1 / cfg.dim)
        dense.weight[0].zero_()
        outputs.weight.zero_()
    initial = dense.weight.detach().clone()
    tied = grouped_embedding_bag(initial, partition, padding_idx=0)
    tied_outputs = nn.Embedding.from_pretrained(outputs.weight.detach().clone(), freeze=False, sparse=True)
    optimizers = [torch.optim.SGD(list(x.parameters()) + list(y.parameters()), lr=cfg.lr)
                  for x, y in ((dense, outputs), (tied, tied_outputs))]
    probability = torch.tensor([counts[w] ** .5 for w in words], dtype=torch.float32)
    probability /= probability.sum()
    snapshots = []
    center_events = np.zeros(len(words), dtype=np.int64)
    started = time.perf_counter()
    checkpoints = {int(v) for v in cfg.checkpoints.split(',')}
    for step in range(cfg.steps + 1):
        if step in checkpoints or step == cfg.steps:
            entry = {'step': step, 'sampled_pairs': int(center_events.sum()),
                     **dynamics_snapshot(dense.weight, initial, partition, grams, offset)}
            with torch.no_grad():
                original = dense(bags) / lengths[:, None]
                grouped = tied(bags, check=False) / lengths[:, None]
            entry['matched_tied_training_aggregate_error'] = vector_metrics(original.numpy(), grouped.numpy())
            entry['maximum_tied_output_weight_error'] = float((outputs.weight - tied_outputs.weight).detach().abs().max())
            snapshots.append(entry)
            print(json.dumps({'order': order, 'seed': seed, 'convention': convention, 'step': step,
                              'cosine': entry['identical_incidence_pairs']['cosine'],
                              'seconds': time.perf_counter() - started}), flush=True)
        if step == cfg.steps:
            break
        batch = pairs[rng.integers(len(pairs), size=cfg.batch_size)]
        center, target = torch.from_numpy(batch[:, 0].copy()), torch.from_numpy(batch[:, 1].copy())
        np.add.at(center_events, batch[:, 0], 1)
        negative = torch.multinomial(probability, cfg.batch_size * cfg.negatives, replacement=True).reshape(cfg.batch_size, cfg.negatives)
        for model, output, optimizer in ((dense, outputs, optimizers[0]), (tied, tied_outputs, optimizers[1])):
            summed = model(bags[center]) if model is dense else model(bags[center], check=False)
            averaged = summed / lengths[center, None]
            hidden = summed + (averaged - summed).detach() if convention == 'legacy' else averaged
            positive = (hidden * output(target)).sum(1)
            logits = (hidden[:, None] * output(negative)).sum(2)
            loss = (-F.logsigmoid(positive) - F.logsigmoid(-logits).sum(1)).sum()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss; retain failed run')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = {'input_rows': dense.weight.detach(), 'output_rows': outputs.weight.detach(),
                  'initial_rows': initial, 'words': words, 'grams': grams, 'bags': bags, 'lengths': lengths,
                  'order': order, 'seed': seed}
    torch.save(checkpoint, directory / 'reference.pt')
    torch.save({'input_rows': tied.expanded_rows().detach(), 'output_rows': tied_outputs.weight.detach()}, directory / 'tied-reference.pt')
    result = {'convention': convention, 'order': order, 'seed': seed, 'partition': partition.summary(),
              'training_seconds': time.perf_counter() - started, 'snapshots': snapshots,
              'sampled_center_events_sha256': hashlib.sha256(center_events.astype('<u8').tobytes()).hexdigest(),
              'sampled_center_events': distribution(center_events),
              'original_input_parameters': nrows * cfg.dim, 'grouped_input_parameters': len(partition.sizes) * cfg.dim,
              'checkpoint_sha256': file_sha256(directory / 'reference.pt')}
    write_json(directory / 'dynamics.json', result)
    return checkpoint, result


def oov_audit(tokens, checkpoint, canonical, partition, limit=4096):
    words, grams = checkpoint['words'], checkpoint['grams']
    offset = len(words) + 1
    lookup = {g: i + offset for i, g in enumerate(grams)}
    vocabulary = set(words)
    frequencies = Counter(w for w in tokens if w not in vocabulary)
    # Deterministic lexical subset, never used to fit or tune the partition.
    selected = sorted(frequencies)[:limit]
    errors, accepted = [], 0
    old = checkpoint['input_rows'].numpy()
    for word in selected:
        bag = [lookup[g] for g in ngrams(word, checkpoint['order']) if g in lookup]
        safe = partition.certifies(bag)
        accepted += int(safe)
        if bag:
            a = old[bag].astype(np.float64).mean(0)
            b = canonical[bag].astype(np.float64).mean(0)
            errors.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)))
    return {'oov_tokens': sum(frequencies.values()), 'oov_types': len(frequencies),
            'audited_oov_types': len(selected), 'certified_oov_types': accepted,
            'rejected_oov_types': len(selected) - accepted,
            'unsafe_unconditional_projection_relative_l2': distribution(errors),
            'scope': 'partial registered n-gram bags for unseen words; not a downstream OOV semantic benchmark'}


def export_variant(rows, representatives, partition, cfg, directory, name, common_bytes):
    path = directory / name
    path.mkdir()
    raw_path, artifact = path / 'staging.f32', path / 'table.men'
    rows.astype('<f4').tofile(raw_path)
    packed = pack_native(raw_path, artifact, cfg.dim * 4, block_rows=16,
                         deduplicate=True, representatives=representatives.tolist(), predictor='xor')
    expected = rows[representatives].astype('<f4').tobytes()
    with NativeRows(artifact, cfg.native_library, cache_bytes=4 << 20) as native:
        actual = native.rows(range(len(rows)))
    if actual != expected:
        raise AssertionError('native bytes differ from declared representative rows')
    raw_path.unlink()
    if partition is not None:
        partition.save(path / 'domain.npz')
    manifest = {'name': name, 'source_row_byte_preserving': name == 'identity',
                'native_decodes_declared_rows_byte_exactly': True,
                'decoded_sha256': hashlib.sha256(actual).hexdigest(),
                'artifact_sha256': file_sha256(artifact), 'storage': packed,
                'contract': ('certified bag aggregates in real arithmetic; FP rounding measured; reject uncertified bags'
                             if partition is not None else 'original rows' if name == 'identity' else 'approximate rows')}
    if partition is not None:
        manifest['partition'] = partition.summary()
    write_json(path / 'manifest.json', manifest)
    manifest['complete_input_output_key_bytes'] = common_bytes + sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    return torch.from_numpy(np.frombuffer(actual, dtype='<f4').reshape(rows.shape).copy()), manifest


def audit_checkpoint(checkpoint, split, cfg, directory):
    words, grams = checkpoint['words'], checkpoint['grams']
    bags, lengths = checkpoint['bags'], checkpoint['lengths']
    rows, outputs = checkpoint['input_rows'], checkpoint['output_rows']
    if not torch.isfinite(rows).all() or not torch.isfinite(outputs).all():
        raise ValueError('nonfinite source checkpoint')
    offset, nrows = len(words) + 1, len(rows)
    partition = fit_partition(bags, nrows, offset)
    canonical = partition.means(rows.numpy())[partition.mapping]
    counts = Counter(split['train'])
    evaluation = {name: evaluation_set(split[name], words, counts, cfg.eval_pairs, cfg.negatives, 42 + (name == 'test'))
                  for name in ('validation', 'test')}
    original_hidden = compose(rows, bags, lengths)
    canonical_hidden = compose(torch.from_numpy(canonical), bags, lengths)
    reference = {name: score(original_hidden, outputs, data) for name, data in evaluation.items()}
    common_keys = json.dumps([words, grams], ensure_ascii=True).encode()
    common_bytes = outputs.numel() * outputs.element_size() + len(common_keys)
    keys = {offset + i: g for i, g in enumerate(grams)}
    identity = np.arange(nrows)
    candidates = [('identity', rows.numpy(), identity, None),
                  ('certified-groups', canonical, partition.representatives(), partition)]
    if cfg.structural_library:
        for name, source, mode, domain in [('original-joint099', rows.numpy(), 'joint', None),
                                            ('canonical-datrie-exact', canonical, 'exact', partition)]:
            with StructuralPlan(source.astype('<f4').tobytes(), cfg.dim, keys, cfg.structural_library, mode=mode) as plan:
                candidates.append((name, source, np.asarray(plan.representatives), domain))
    variants = []
    for name, source, mapping, domain in candidates:
        decoded, result = export_variant(source, mapping, domain, cfg, directory, name, common_bytes)
        hidden = compose(decoded, bags, lengths)
        result['aggregate_fidelity'] = vector_metrics(original_hidden.numpy(), hidden.numpy())
        result['source_row_fidelity'] = vector_metrics(rows.numpy(), decoded.numpy())
        result['quality'] = {split_name: {'baseline': summarize(reference[split_name]),
                              'candidate': summarize(score(hidden, outputs, data)),
                              'delta': deltas(score(hidden, outputs, data), reference[split_name]),
                              'token_coverage': data['token_coverage'], 'evaluation_sha256': data['sha256']}
                             for split_name, data in evaluation.items()}
        if domain is not None:
            with CertifiedBagRows(directory / name, cfg.native_library) as native:
                native.table.set_cache_bytes(4 << 20)
                started = time.perf_counter()
                compiled = native.compile(bags.tolist(), padding_idx=0)
                compile_seconds = time.perf_counter() - started
                samples = []
                for _ in range(6):
                    started = time.perf_counter()
                    aggregates = native.evaluate(compiled, mean=True)
                    samples.append(time.perf_counter() - started)
                result['compiled_native_bags'] = {**compiled.summary(), 'compile_seconds': compile_seconds,
                    'first_query_seconds': samples[0], 'repeated_query_seconds': samples[1:],
                    'aggregate_fidelity': vector_metrics(original_hidden.numpy(), aggregates),
                    'limitations': 'all registered words in one batch; compilation excluded from query samples; shared CPU and warm OS pages'}
                result['complete_input_output_key_and_plan_bytes'] = result['complete_input_output_key_bytes'] + compiled.summary()['plan_bytes']
        variants.append(result)
    # Negative control reuses learned rows and group sizes; it is not new training data.
    rng = np.random.default_rng(checkpoint['seed'] + 123)
    permutation = np.arange(nrows)
    permutation[offset:] = rng.permutation(permutation[offset:])
    random_partition = BagPartition(partition.mapping[permutation], 'negative-control', len(words))
    random_rows = random_partition.means(rows.numpy())[random_partition.mapping]
    random_hidden = compose(torch.from_numpy(random_rows), bags, lengths)
    return {'order': checkpoint['order'], 'seed': checkpoint['seed'], 'partition': partition.summary(),
            'source_input_sha256': hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
            'source_output_sha256': hashlib.sha256(outputs.numpy().tobytes()).hexdigest(),
            'original_full_vector_payload_bytes': (rows.numel() + outputs.numel()) * 4,
            'common_output_and_keys_bytes': common_bytes,
            'canonical_aggregate_fidelity': vector_metrics(original_hidden.numpy(), canonical_hidden.numpy()),
            'random_group_negative_control_aggregate_fidelity': vector_metrics(original_hidden.numpy(), random_hidden.numpy()),
            'variants': variants,
            'oov': {name: oov_audit(split[name], checkpoint, canonical, partition) for name in ('validation', 'test')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--structural-library', type=Path)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--max-orders', default='4,8,12')
    parser.add_argument('--seeds', default='3,7,11')
    parser.add_argument('--conventions', default='legacy,average')
    parser.add_argument('--checkpoints', default='0,100,500,2000,4000')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--vocab-limit', type=int, default=1200)
    parser.add_argument('--min-count', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--eval-pairs', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=.05)
    cfg = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('set CUDA_VISIBLE_DEVICES to empty; this study is CPU only')
    if min(cfg.dim, cfg.steps, cfg.vocab_limit, cfg.min_count, cfg.batch_size, cfg.negatives, cfg.eval_pairs) < 1 or cfg.lr <= 0:
        parser.error('positive dimensions, steps and learning rate required')
    orders, seeds = [int(v) for v in cfg.max_orders.split(',')], [int(v) for v in cfg.seeds.split(',')]
    conventions = cfg.conventions.split(',')
    if min(orders) < 2 or any(v not in ('legacy', 'average') for v in conventions):
        parser.error('order >= 2 and legacy/average conventions required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    raw = cfg.corpus.read_bytes()
    split = split_tokens(raw.decode('utf-8'))
    report = {'category': 'finite public-text CPU SGNS; aggregate-domain equivalence and training dynamics',
              'source_commit': os.environ.get('MEFT_SOURCE_COMMIT') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
              'torch': torch.__version__, 'numpy': np.__version__, 'os': platform.platform(),
              'cpu': platform.processor(), 'threads': 1, 'device': 'cpu',
              'corpus_sha256': hashlib.sha256(raw).hexdigest(),
              'split_token_sha256': {k: hashlib.sha256('\n'.join(v).encode()).hexdigest() for k, v in split.items()},
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
              'limitations': ['Identical fixed bag-incidence columns, not arbitrary prefixes or LLM memory.',
                              'Known-word SGNS evaluation excludes OOV centers/targets; OOV bags audited and rejected separately.',
                              'Canonicalization changes individual vectors; real-arithmetic aggregate identity is not byte identity.',
                              'Sparse SGD without decay/regularization; finite training, no convergence claim.',
                              'One small English public corpus; no GPU/whole-LLM task result; monetary/energy cost unmeasured.',
                              'Corpus license not independently verified; publish aggregate results/hashes only.'],
              'runs': []}
    # Freeze protocol before seeing checkpoint/training outcomes.
    write_json(cfg.output_dir / 'protocol.json', {k: v for k, v in report.items() if k != 'runs'})
    for order in orders:
        for seed in seeds:
            for convention in (['cached'] if cfg.checkpoint_root else conventions):
                directory = cfg.output_dir / f'{convention}-order-{order}-seed-{seed}'
                if cfg.checkpoint_root:
                    directory.mkdir()
                    source = cfg.checkpoint_root / f'order-{order}-seed-{seed}' / 'reference.pt'
                    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
                    if checkpoint['order'] != order or checkpoint['seed'] != seed:
                        raise ValueError('checkpoint order/seed mismatch')
                    if checkpoint['input_rows'].shape[1] != cfg.dim:
                        raise ValueError('checkpoint dimension differs from declared --dim')
                    dynamics = {'source_checkpoint_sha256': file_sha256(source)}
                else:
                    checkpoint, dynamics = train_dynamics(split, cfg, order, seed, convention, directory)
                run = audit_checkpoint(checkpoint, split, cfg, directory)
                run['dynamics'] = dynamics
                write_json(directory / 'report.json', run)
                report['runs'].append(run)
                write_json(cfg.output_dir / 'report.json', report)
    report['status'] = 'completed'
    write_json(cfg.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
