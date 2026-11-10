"""Paired finite application comparisons; never reselects on held-out data."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tree_node_quality import load, write

GROUPS = ('rewrite', 'paraphrase', 'neighborhood')


def case_matrix(cases, split='test'):
    cases = sorted((c for c in cases if c['split'] == split), key=lambda c: c['case_id'])
    ids = [c['case_id'] for c in cases]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Unique nonempty official paired cases required')
    return ids, np.asarray([[sum(v['success'] for v in c['groups'][g]) / len(c['groups'][g])
                             for g in GROUPS] for c in cases], dtype=np.float64)


def metric_vector(success):
    success = np.asarray(success)
    denominator = np.divide(1., success, out=np.full_like(success, np.inf), where=success > 0).sum(-1)
    harmonic = 3. / denominator
    return np.concatenate((harmonic[..., None], success), axis=-1)


def hierarchical_paired(sources, candidates, replicates=2000, seed=42):
    """Resample seed clusters and the same case indices within each pair."""
    if len(sources) != 3 or len(candidates) != 3:
        raise ValueError('All three registered seeds required')
    n = sources[0].shape[0]
    if any(s.shape != (n, 3) for s in [*sources, *candidates]):
        raise ValueError('Paired official case matrices differ')
    observed = metric_vector(np.mean([s.mean(0) for s in candidates], 0)) - metric_vector(np.mean([s.mean(0) for s in sources], 0))
    rng = np.random.default_rng(seed); samples = []
    for _ in range(replicates):
        chosen = rng.integers(0, 3, 3)
        a = []; b = []
        for index in chosen:
            ids = rng.integers(0, n, n)
            a.append(sources[index][ids].mean(0)); b.append(candidates[index][ids].mean(0))
        samples.append((metric_vector(np.mean(b, 0)) - metric_vector(np.mean(a, 0))).tolist())
    return {'metrics': ['harmonic_score', *GROUPS], 'observed_candidate_minus_source': observed.tolist(),
            'paired_95_percent_intervals': np.quantile(samples, [.025, .975], axis=0).T.tolist(),
            'replicates': replicates, 'seed': seed, 'seed_clusters': 3, 'cases_per_seed': n, 'samples': samples,
            'scope': 'Pointwise three-seed and official case resampling; fixed checkpoints and calibration-selected candidates. No independent-task or universal guarantee.'}


def analyze_llm(root, output):
    frozen = load(root / 'llm/selection.json'); summaries = []
    source = {}; variants = {}; costs = {}; source_costs = {}
    for selected in frozen['models']:
        name = selected['name']; folder = root / 'llm/models' / name
        grid = {v['build']['name']: v for v in load(folder / 'grid.json.gz')}
        controls = {c['name']: c for c in load(folder / 'controls.json.gz')}
        raw = grid['raw-fp32']; source_costs[name] = raw['complete_serving_bytes']
        ids, reference = case_matrix(load(root / raw['quality']['test']['public_case_scores']))
        source[name] = reference
        model_summary = {'model': name, 'source_complete_serving_bytes': source_costs[name], 'choices': []}
        for choice in selected['choices']:
            if choice['name'] is None:
                model_summary['choices'].append({**choice, 'test_status': 'no_calibration_qualified_candidate'})
                continue
            entry = grid[choice['name']]
            actual_ids, actual = case_matrix(load(root / entry['quality']['test']['public_case_scores']))
            if actual_ids != ids:
                raise ValueError('Selected model case pairing differs')
            key = (choice['budget_pp'], choice['family'])
            variants.setdefault(key, {})[name] = actual
            costs.setdefault(key, {})[name] = entry['complete_serving_bytes']
            difference = metric_vector(reference.mean(0)) - metric_vector(actual.mean(0))
            model_summary['choices'].append({**choice, 'test_source_minus_candidate': difference.tolist(),
                'test_all_four_metrics_within_original_budget': bool(np.all(difference <= choice['budget_pp'] / 100)),
                'saving_vs_source_fraction': 1. - entry['complete_serving_bytes'] / source_costs[name]})
        for control_name in ('head-constant', 'adapter', 'hash'):
            control = controls[control_name]
            cases_file = control.get('public_native_oov_case_scores', control['public_case_scores']) if control_name == 'hash' else control['public_case_scores']
            actual_ids, actual = case_matrix(load(root / cases_file))
            if actual_ids != ids:
                raise ValueError('Independent control case pairing differs')
            variants.setdefault(('independent', control_name), {})[name] = actual
            report = control['report']
            costs.setdefault(('independent', control_name), {})[name] = (report['delta']['bytes'] + control['private_report_bytes'] if control_name != 'hash' else report['identity_bundle_bytes'])
        summaries.append(model_summary)
    comparisons = []
    for key, points in sorted(variants.items(), key=lambda p: str(p[0])):
        for model in ('qwen06', 'qwen35'):
            names = [f'{model}-exact-s{seed}' for seed in (3, 7, 11)]
            if not all(name in points for name in names):
                comparisons.append({'model': model, 'choice': list(key), 'status': 'some_seeds_have_no_calibration_qualified_candidate',
                                    'available_seeds': [name for name in names if name in points]})
                continue
            result = hierarchical_paired([source[name] for name in names], [points[name] for name in names])
            samples = result.pop('samples')
            file = 'llm-intervals/' + model + '-' + str(key[0]) + '-' + str(key[1]) + '.json.gz'
            write(output / file, {'samples': samples})
            comparisons.append({'model': model, 'choice': list(key), 'status': 'complete', **result,
                'samples_file': file, 'candidate_bytes_by_seed': [costs[key][name] for name in names],
                'source_bytes_by_seed': [source_costs[name] for name in names]})
    write(output / 'llm-summary.json', summaries)
    write(output / 'llm-paired-comparisons.json', comparisons)


def analyze_long(root, output):
    frozen = load(root / 'long/selection.json'); training = load(root / 'long/training-selection.json')
    chosen_sources = {g['chosen'] for g in training['groups']}; result = []
    for selected in frozen['models']:
        folder = root / 'long/models' / selected['name']
        grid = {v['build']['name']: v for v in load(folder / 'grid.json.gz')}
        raw_size = grid['raw-fp32']['complete_serving_bytes']
        choices = []
        for choice in selected['choices']:
            if choice['name'] is None:
                choices.append(choice); continue
            entry = grid[choice['name']]
            choices.append({**choice, 'saving_vs_raw_fp32_fraction': 1. - entry['complete_serving_bytes'] / raw_size,
                'test_quality': entry['audits'][str(choice['cached'])]['report']['quality']['test'],
                'confirmation_quality': entry['audits'][str(choice['cached'])]['report']['quality']['confirmation']})
        result.append({'model': selected['name'], 'validation_chosen_training_point': selected['name'] in chosen_sources,
                       'source_complete_serving_bytes': raw_size, 'choices': choices})
    write(output / 'long-summary.json.gz', result)
    write(output / 'training-choice-summary.json', training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    analyze_llm(args.evidence, args.output); analyze_long(args.evidence, args.output)
    write(args.output / 'report.json', {'status': 'complete', 'selection_rule': 'Frozen calibration/validation choices only; test and confirmation never reselect',
                                      'llm_models': 6, 'long_models': 108, 'replicates': 2000})


if __name__ == '__main__':
    main()
