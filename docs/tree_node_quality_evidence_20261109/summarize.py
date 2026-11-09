"""Derive paper-facing summaries from immutable numeric reports (stdlib)."""
import json
from pathlib import Path
import statistics

from verify import ROOT, load, gate, resolve_evidence


def summarize(root=ROOT):
    selection = load(root / 'selection.json')
    records, comparisons, controls, query_controls = [], [], [], []
    for model in selection['models']:
        folder = root / 'models' / model['name']
        metadata = load(folder / 'metadata.json')
        grid = {g['build']['name']: g for g in load(folder / 'grid.json.gz')}
        timings = {(r['name'], r['cached_known_words'], r['index_kind'], r['native_cache_bytes']): r
                   for r in load(folder / 'retimings.json.gz')}
        ci = {(c['name'], c['cached'], c['split']): c for c in load(folder / 'selected-intervals.json.gz')}
        for (name, cached, index, cache), t in sorted(timings.items()):
            ref = timings['raw-fp32', cached, index, 0]
            query_controls.append({'model': model['name'], 'convention': metadata['convention'],
                'order': metadata['order'], 'seed': metadata['seed'], 'name': name,
                'family': grid[name]['build']['family'], 'cached': cached,
                'index': index, 'native_cache_bytes': cache,
                'complete_serving_bytes': grid[name]['complete_serving_bytes']
                    - metadata['index_bytes']['datrie'] + metadata['index_bytes'][index],
                'load_seconds': t['load_seconds'], 'query_rss_bytes': t['memory_after_query']['rss_bytes'],
                'query_anonymous_bytes': t['memory_after_query']['anonymous_bytes'],
                'complete_query_p50_ms': {s: t['query'][s]['complete_p50_ms'] for s in ('known', 'oov')},
                'query_speedup_to_matched_raw_fp32': {s: ref['query'][s]['complete_p50_ms'] / t['query'][s]['complete_p50_ms'] for s in ('known', 'oov')}})
        original_bytes = grid['raw-fp32']['complete_serving_bytes']
        for choice in model['choices']:
            if not choice['name']:
                records.append({'model': model['name'], **choice, 'eligible': False})
                continue
            name, cached, budget = choice['name'], choice['cached'], choice['budget']
            g = grid[name]
            t = timings[name, cached, 'datrie', 0]
            ref = timings['raw-fp32', cached, 'datrie', 0]
            record = {'model': model['name'], 'convention': metadata['convention'],
                'order': metadata['order'], 'seed': metadata['seed'], **choice,
                'eligible': True, 'complete_serving_bytes': g['complete_serving_bytes'],
                'raw_fp32_complete_bytes': original_bytes,
                'complete_fraction_of_raw_fp32': g['complete_serving_bytes'] / original_bytes,
                'validation': g['build']['validation'][str(cached)],
                'audit': g['audits'][str(cached)]['quality'],
                'test_budget_pass': gate(g['audits'][str(cached)]['quality']['test'], budget),
                'confirmation_budget_pass': gate(g['audits'][str(cached)]['quality']['confirmation'], budget),
                'confirmation_intervals': {s: {k: v for k, v in value.items() if k != 'samples'}
                    for s, value in ci[name, cached, 'confirmation']['strata'].items()},
                'complete_query_p50_ms': {s: t['query'][s]['complete_p50_ms'] for s in ('known', 'oov')},
                'query_ratio_to_raw_fp32': {s: t['query'][s]['complete_p50_ms'] / ref['query'][s]['complete_p50_ms'] for s in ('known', 'oov')},
                'query_rss_bytes': t['memory_after_query']['rss_bytes']}
            records.append(record)
        for name in ('constant-zero', 'constant-mean', 'constant-weighted-mean'):
            for cached in (0, 1):
                g = grid[name]
                controls.append({'model': model['name'], 'convention': metadata['convention'],
                    'order': metadata['order'], 'seed': metadata['seed'], 'name': name, 'cached': cached,
                    'complete_serving_bytes': g['complete_serving_bytes'],
                    'validation_primary_pass': gate(g['build']['validation'][str(cached)], .001),
                    'quality': g['audits'][str(cached)]['quality']})
        primary = [r for r in records if r['model'] == model['name'] and r['budget'] == .001 and r['eligible']]
        for cached in (0, 1):
            candidates = [r for r in primary if r['cached'] == cached]
            by_family = {r['family']: r for r in candidates}
            for tree in ('tree-datrie', 'tree-reference'):
                if tree not in by_family:
                    continue
                for target, families in (('raw', ('raw',)), ('native-identity', ('native-identity',)),
                        ('best-faiss-including-dedup', ('faiss', 'faiss-dedup'))):
                    others = [by_family[f] for f in families if f in by_family]
                    if not others:
                        continue
                    a = by_family[tree]; b = min(others, key=lambda r: (r['complete_serving_bytes'], r['name']))
                    comparisons.append({'model': model['name'], 'convention': metadata['convention'],
                        'order': metadata['order'], 'seed': metadata['seed'], 'cached': cached,
                        'tree_family': tree, 'tree_name': a['name'], 'target': target, 'baseline_name': b['name'],
                        'additional_complete_saving': 1 - a['complete_serving_bytes'] / b['complete_serving_bytes'],
                        'known_complete_speed_ratio': b['complete_query_p50_ms']['known'] / a['complete_query_p50_ms']['known'],
                        'oov_complete_speed_ratio': b['complete_query_p50_ms']['oov'] / a['complete_query_p50_ms']['oov'],
                        'both_confirmation_budget_pass': a['confirmation_budget_pass'] and b['confirmation_budget_pass']})
    mechanisms = []
    for path in sorted(resolve_evidence(root / 'points').glob('*/report.json')):
        r = load(path); relations = load(path.parent / 'conditional.json.gz')['relations']
        mechanisms.append({'name': path.parent.name, 'convention': r['convention'], 'order': r['order'],
            'seed': r['seed'], 'scale': r['ngram_initialization_scale'], 'steps': r['point']['steps'],
            'update_to_initial_norm_ratio': r['residuals']['update_to_initial_norm_ratio'],
            'common_update_dispersion_relative': r['residuals']['group_update_dispersion_relative'],
            'initial_residual_norm': r['residuals']['initial_residual_norm'],
            'final_residual_norm': r['residuals']['final_residual_norm'],
            'source_confirmation_quality': r['source_quality']['confirmation'],
            'parent_cosine99_fraction': relations[0]['source']['ge_0_99_pairs'] / relations[0]['sampled_pairs'],
            'path_cosine99_fraction': relations[1]['source']['ge_0_99_pairs'] / relations[1]['sampled_pairs'],
            'conditional_associations': {k: v['weighted_spearman'] for k, v in relations[1]['ancestry_diagnostics'].items()
                                         if isinstance(v, dict) and 'weighted_spearman' in v},
            'variants': {name: {'native_table_saving': 1 - v['row_bundle_bytes'] / r['native_identity_row_bundle_bytes'],
                'merged_fraction': v['merged_rows'] / r['shape'][0],
                'confirmation_quality': v['quality']['0']['confirmation']}
                         for name, v in r['variants'].items()}})
    mechanism_groups = []
    for convention in ('legacy', 'average'):
        for order in (4, 8, 12):
            for scale in (1., .1):
                for steps in (2000, 8000, 32000):
                    group = [r for r in mechanisms if (r['convention'], r['order'], r['scale'], r['steps']) == (convention, order, scale, steps)]
                    assert len(group) == 3
                    mechanism_groups.append({'convention': convention, 'order': order, 'scale': scale, 'steps': steps,
                        'median_update_to_initial_norm_ratio': statistics.median(r['update_to_initial_norm_ratio'] for r in group),
                        'median_parent_cosine99_fraction': statistics.median(r['parent_cosine99_fraction'] for r in group),
                        'median_path_cosine99_fraction': statistics.median(r['path_cosine99_fraction'] for r in group),
                        'median_source_confirmation_loss': statistics.median(r['source_confirmation_quality']['all']['sgns_loss_per_pair'] for r in group),
                        'median_source_confirmation_discrimination': statistics.median(r['source_confirmation_quality']['all']['context_discrimination'] for r in group),
                        'variants': {v: {'median_native_table_saving': statistics.median(r['variants'][v]['native_table_saving'] for r in group),
                            'minimum_native_table_saving': min(r['variants'][v]['native_table_saving'] for r in group),
                            'maximum_native_table_saving': max(r['variants'][v]['native_table_saving'] for r in group),
                            'maximum_confirmation_loss_increase': max(r['variants'][v]['confirmation_quality']['all']['loss_relative_change'] for r in group),
                            'maximum_confirmation_oov_loss_increase': max(r['variants'][v]['confirmation_quality']['oov_nonempty']['loss_relative_change'] for r in group)}
                                     for v in ('datrie', 'reference')}})
    primary = [r for r in records if r['budget'] == .001 and r['eligible']]
    tally = {f: {'selected': sum(r['family'] == f for r in primary),
        'test_failures': sum(r['family'] == f and not r['test_budget_pass'] for r in primary),
        'confirmation_failures': sum(r['family'] == f and not r['confirmation_budget_pass'] for r in primary)}
             for f in ('raw', 'dense', 'constant', 'native-identity', 'tree-datrie', 'tree-reference', 'faiss', 'faiss-dedup')}
    return {'primary_budget': .001, 'frozen_choices': records, 'primary_family_tally': tally,
        'primary_comparisons': comparisons, 'necessity_controls': controls, 'query_controls': query_controls,
        'mechanism_points': mechanisms, 'mechanism_groups': mechanism_groups,
        'scope': 'Finite frozen SGNS context task and shared-CPU complete string embedding queries; no LLM or universal dominance claim.'}


if __name__ == '__main__':
    (ROOT / 'summary.json').write_text(json.dumps(summarize(), indent=2, sort_keys=True, allow_nan=False) + '\n')
