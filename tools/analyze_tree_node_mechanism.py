"""Analyze every point in the fixed initialization/exposure factorial on CPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .tree_node_quality import (array_digest, bootstrap, digest, environment,
    load, paired_blocks, block_summary, plan_for, strata_masks, write)


def residual_diagnostics(initial, final, partition):
    initial, final = initial.astype(np.float64), final.astype(np.float64)
    updates = final - initial
    size = partition.sizes
    def means(rows):
        total = np.zeros((len(size), rows.shape[1]), dtype=np.float64)
        np.add.at(total, partition.mapping, rows)
        return total / size[:, None]
    initial_mean, final_mean, update_mean = means(initial), means(final), means(updates)
    init_residual = initial - initial_mean[partition.mapping]
    final_residual = final - final_mean[partition.mapping]
    update_residual = updates - update_mean[partition.mapping]
    return {'groups': len(size), 'rows_in_nonsingleton_groups': int(size[size > 1].sum()),
        'initial_norm': float(np.linalg.norm(initial)), 'update_norm': float(np.linalg.norm(updates)),
        'initial_residual_norm': float(np.linalg.norm(init_residual)),
        'final_residual_norm': float(np.linalg.norm(final_residual)),
        'group_update_dispersion_relative': float(np.linalg.norm(update_residual) / max(np.linalg.norm(updates), 1e-30)),
        'residual_change_relative_to_initial': float(np.linalg.norm(final_residual - init_residual) / max(np.linalg.norm(init_residual), 1e-30)),
        'update_to_initial_norm_ratio': float(np.linalg.norm(updates) / max(np.linalg.norm(initial), 1e-30))}


def analyze_point(job_path):
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from .analyze_conditional_similarity import analyze, metadata
    from .benchmark_faiss_cotraining import evaluate, objective
    from .benchmark_cosine_merge import structural_merge, fidelity
    from .cotraining_groups import BagPartition
    from .quantized_memory import pack_quantized, QuantizedRows
    from .structural_native import StructuralPlan
    from .benchmark_heldout_compression import split_tokens
    job = load(job_path); point = Path(job['checkpoint']); common = Path(job['common'])
    source = torch.load(point, map_location='cpu', weights_only=True)
    assert digest(point) == job['point_record']['checkpoint_sha256']
    offset = len(source['words']) + 1
    rows, initial = source['input_rows'].numpy()[offset:], source['initial_rows'].numpy()[offset:]
    assert array_digest(source['input_rows'].numpy()) == job['point_record']['input_row_sha256']
    settings = dict(load(Path(job['training_root']) / 'report.json')['settings'], steps=0)
    text = ' '.join(split_tokens((Path(job['training_root']) / 'corpus.txt').read_text())['train'])
    words, keys, counts, _, signatures, _ = metadata(text, source['order'], settings, source['seed'])
    assert words == source['words'] and keys == source['grams']
    events = np.load(point.parent / 'center-events.npy')
    assert array_digest(events.astype('<u8')) == job['point_record']['sampled_center_events_sha256']
    exposures = np.zeros(len(rows), dtype=np.int64)
    training_bags = [[int(i) - offset for i in bag if i >= offset] for bag in source['bags'].numpy()]
    for event, bag in zip(events, training_bags):
        np.add.at(exposures, bag, event)
    partition = BagPartition.fit(training_bags, len(rows))
    relation_results = analyze(rows, keys, counts, ('parent', 'ancestor'),
        initial=initial, signatures=signatures, exposures=exposures)
    out = Path(job['output']); out.mkdir(parents=True)
    write(out / 'conditional.json.gz', {'relations': relation_results})
    result = {'environment': environment(), 'torch': torch.__version__, 'job_sha256': digest(job_path),
        'point': job['point_record'], 'order': source['order'], 'seed': source['seed'],
        'convention': job['convention'], 'ngram_initialization_scale': job['scale'],
        'source_checkpoint_sha256': digest(point), 'shape': list(rows.shape),
        'conditional_report_sha256': digest(out / 'conditional.json.gz'),
        'residuals': residual_diagnostics(initial, rows, partition), 'variants': {}}
    lexical_bags = load(common / 'lexical-bags.json')
    word_ids = np.load(common / 'lexical-word-ids.npy')
    nonempty = np.load(common / 'lexical-nonempty.npy')
    word_rows = source['input_rows'].numpy()[1:offset]
    class ArrayRows:
        def __init__(self, matrix):
            self.matrix = matrix
            self.mapping = None
            self.metadata = {'rows': len(matrix), 'dim': matrix.shape[1]}
        def rows(self, ids):
            return self.matrix[ids]
    oracle = ArrayRows(rows)
    plan = plan_for(lexical_bags, word_ids, oracle)
    hidden = evaluate(oracle, plan, word_rows, batch_bags=64)
    outputs = source['output_rows'].numpy()
    evaluations = {split: dict(np.load(common / (split + '-evaluation.npz'))) for split in ('validation', 'test', 'confirmation')}
    baseline = {split: objective(hidden, outputs, e) for split, e in evaluations.items()}
    result['source_quality'] = {split: {name: block_summary(paired_blocks(score, score, e['positions'], mask))
        for name, mask in strata_masks(e, word_ids, nonempty).items()}
        for (split, e), score in zip(evaluations.items(), baseline.values())}
    baseline_dir = out / 'identity'
    identity = pack_quantized(rows, baseline_dir, 'fp32', native=True, max_block_bytes=8192)
    native_identity_bytes = sum(p.stat().st_size for p in baseline_dir.iterdir())
    common_bytes = load(common / 'metadata.json')['common_model_bytes']
    result['native_identity_row_bundle_bytes'] = native_identity_bytes
    result['native_identity_model_bytes'] = common_bytes + native_identity_bytes
    mappings = {}
    with StructuralPlan(rows.tobytes(), rows.shape[1], dict(enumerate(keys)), job['structural_library'],
                        mode='joint', cosine=.99, norm_ratio=.99) as native_plan:
        mappings['datrie'] = (np.asarray(native_plan.representatives), native_plan.row_order)
    reference = structural_merge(rows, keys, .99, .99)
    order = [i for i in sorted(range(len(keys)), key=lambda i: keys[i]) if reference[i] == i] + [i for i in range(len(keys)) if reference[i] != i]
    mappings['reference'] = (reference, order)
    for name, (mapping, row_order) in mappings.items():
        directory = out / name
        packed = pack_quantized(rows, directory, 'fp32', native=True, representatives=mapping.tolist(),
                                row_order=row_order, max_block_bytes=8192)
        with QuantizedRows(directory, job['native_library'], cache_bytes=0) as reader:
            for start in range(0, len(rows), 4096):
                ids = np.arange(start, min(start + 4096, len(rows)))
                assert reader.gather(ids).tobytes() == rows[mapping[ids]].tobytes()
        changed = mapping != np.arange(len(mapping))
        variant = {'merged_rows': int(changed.sum()), 'retained_rows': int((~changed).sum()),
            'mapping_sha256': array_digest(mapping.astype('<u8')), 'fidelity': fidelity(rows, rows[mapping], counts),
            'row_bundle_bytes': sum(p.stat().st_size for p in directory.iterdir()),
            'native_reconstruction_byte_exact_to_approximation': True, 'quality': {}}
        variant['model_bytes'] = common_bytes + variant['row_bundle_bytes']
        candidate_hidden = evaluate(ArrayRows(rows[mapping]), plan, word_rows, batch_bags=64)
        for cached in (0, 1):
            actual = candidate_hidden.copy()
            if cached:
                actual[:len(words)] = hidden[:len(words)]
            variant['quality'][str(cached)] = {}
            for split, e in evaluations.items():
                scores = objective(actual, outputs, e)
                blocks = {k: paired_blocks(scores, baseline[split], e['positions'], mask)
                    for k, mask in strata_masks(e, word_ids, nonempty).items()}
                variant['quality'][str(cached)][split] = {k: block_summary(b) for k, b in blocks.items()}
                write(out / f'{name}-{split}-cache{cached}-blocks.json.gz', blocks)
                if not cached and split in ('test', 'confirmation'):
                    write(out / f'{name}-{split}-intervals.json.gz',
                          {k: bootstrap(blocks[k]) for k in ('all', 'oov_nonempty')})
        result['variants'][name] = variant
    result['status'] = 'complete'
    write(out / 'report.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', type=Path)
    parser.add_argument('--frontier', type=Path)
    parser.add_argument('--training-root', type=Path)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--structural-library', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int, default=24)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU only; empty CUDA_VISIBLE_DEVICES required')
    if args.worker:
        analyze_point(args.worker); return
    source = load(args.mechanism / 'report.json'); assert source['status'] == 'complete'
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'environment': environment(), 'source_factorial_report_sha256': digest(args.mechanism / 'report.json'),
              'jobs': [], 'complete': [], 'failures': []}
    for t in source['trajectories']:
        trajectory_dir = args.mechanism / t['name']; trajectory = load(trajectory_dir / 'report.json')
        assert digest(trajectory_dir / 'report.json') == t['report_sha256']
        for p in trajectory['checkpoints']:
            name = t['name'].replace('/', '-') + f'-step-{p["steps"]}'
            job = {'checkpoint': str(trajectory_dir / f'step-{p["steps"]}' / 'reference.pt'),
                'point_record': p, 'convention': trajectory['convention'], 'scale': trajectory['ngram_initialization_scale'],
                'training_root': str(args.training_root),
                'common': str(args.frontier / f'{trajectory["convention"]}-order-{trajectory["order"]}-seed-{trajectory["seed"]}' / 'common'),
                'native_library': str(args.native_library), 'structural_library': str(args.structural_library),
                'output': str(args.output / 'points' / name)}
            path = args.output / 'jobs' / (name + '.json'); write(path, job)
            report['jobs'].append(str(path))
    assert len(report['jobs']) == 108
    def execute(path):
        log = path.with_suffix('.log')
        with log.open('w') as f:
            result = subprocess.run([sys.executable, '-m', 'tools.analyze_tree_node_mechanism', '--worker', str(path)], stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{path.stem}: worker exit {result.returncode}')
        output = Path(load(path)['output']) / 'report.json'
        return {'name': path.stem, 'report_sha256': digest(output)}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(execute, Path(path)): path for path in report['jobs']}
        for future in as_completed(futures):
            try:
                report['complete'].append(future.result())
            except Exception as error:
                report['failures'].append({'job': futures[future], 'error': repr(error)})
            write(args.output / 'report.json', report)
            print(json.dumps({'complete_points': len(report['complete']), 'failures': len(report['failures'])}), flush=True)
    report['status'] = 'complete' if not report['failures'] and len(report['complete']) == 108 else 'failed'
    write(args.output / 'report.json', report)
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
