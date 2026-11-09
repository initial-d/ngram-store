"""Vector-blind structural pairs, matched controls and conditional diagnostics.

Finite-checkpoint descriptive statistics, not a universal frequency law or a
causal estimate. Character and token-head coordinate spaces stay separate.
"""
from bisect import bisect_right
from collections import defaultdict
from itertools import combinations
import math

import numpy as np

SEED = 20261109
LIMIT = 50000


def frequency_bins(counts):
    x = np.asarray(counts, dtype=np.int64)
    if (x < 1).any():
        raise ValueError('positive occurrence counts required')
    return np.array([int(v).bit_length() - 1 for v in x], dtype=np.int64)


def related(a, b):
    """Immediate ancestry or equal-length length-1 prefix/suffix siblings."""
    if a == b:
        return True
    if len(a) == len(b):
        return a[:-1] == b[:-1] or a[1:] == b[1:]
    if len(a) != len(b):
        short, long = (a, b) if len(a) < len(b) else (b, a)
        return short == long[:len(short)] or short == long[-len(short):]
    return False


def unrank_pair(n, k):
    if not 0 <= k < n * (n - 1) // 2:
        raise ValueError('pair index out of range')
    prefix = lambda a: a * (2 * n - a - 1) // 2
    a = ((2 * n - 1) - math.isqrt((2 * n - 1) ** 2 - 8 * k)) // 2
    while prefix(a) > k:
        a -= 1
    while a + 1 < n and prefix(a + 1) <= k:
        a += 1
    return a, a + 1 + k - prefix(a)


def structural_pairs(keys, relation, limit=LIMIT, seed=SEED):
    if len(set(keys)) != len(keys) or limit < 1:
        raise ValueError('unique keys and positive sample limit required')
    if relation in ('parent', 'ancestor'):
        lookup = {key: i for i, key in enumerate(keys)}
        pairs = sorted({(lookup[parent], i) for i, key in enumerate(keys)
                        for length in ([len(key) - 1] if relation == 'parent' else range(2, len(key)))
                        for parent in (key[:length], key[-length:]) if parent in lookup})
        # Parent enumeration is small and complete; the cap applies to siblings.
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), len(pairs)
    if relation not in ('prefix', 'suffix'):
        raise ValueError('unknown relation')
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key[:-1] if relation == 'prefix' else key[1:]].append(i)
    groups = [groups[k] for k in sorted(groups) if len(groups[k]) > 1]
    ends, total = [], 0
    for group in groups:
        total += len(group) * (len(group) - 1) // 2
        ends.append(total)
    if total <= limit:
        return np.asarray([p for g in groups for p in combinations(g, 2)], dtype=np.int64).reshape(-1, 2), total
    rng = np.random.default_rng(seed)
    selected = set()
    while len(selected) < limit:
        selected.add(int(rng.integers(total)))
    pairs = []
    for k in sorted(selected):
        group_index = bisect_right(ends, k)
        group = groups[group_index]
        local = k - (ends[group_index - 1] if group_index else 0)
        a, b = unrank_pair(len(group), local)
        pairs.append((group[a], group[b]))
    return np.asarray(pairs, dtype=np.int64), total


def matched_controls(keys, counts, pairs, seed=SEED, attempts=64):
    """Keep anchor and match replacement length/count bin without reading rows."""
    bins = frequency_bins(counts)
    pools = defaultdict(list)
    for i, key in enumerate(keys):
        pools[(len(key), int(bins[i]))].append(i)
    pools = {k: np.asarray(v, dtype=np.int64) for k, v in pools.items()}
    rng = np.random.default_rng(seed)
    result = np.full(len(pairs), -1, dtype=np.int64)
    for position, (a, b) in enumerate(pairs):
        pool = pools[(len(keys[b]), int(bins[b]))]
        candidates = rng.choice(pool, min(attempts, len(pool)), replace=False)
        best, distance = -1, float('inf')
        for c in candidates:
            if c == b or related(keys[a], keys[c]):
                continue
            current = abs(math.log2(int(counts[c]) / int(counts[b])))
            if current < distance:
                best, distance = int(c), current
        result[position] = best
    return result


def pair_metrics(rows, pairs):
    rows = np.asarray(rows)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError('finite dense source rows required')
    cosine = np.full(len(pairs), np.nan)
    ratio = np.full(len(pairs), np.nan)
    error = np.full(len(pairs), np.nan)
    for start in range(0, len(pairs), 4096):
        q = pairs[start:start + 4096]
        a, b = rows[q[:, 0]].astype(np.float64), rows[q[:, 1]].astype(np.float64)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        valid = (na > 0) & (nb > 0)
        sl = slice(start, start + len(q))
        cosine[sl][valid] = np.clip(np.einsum('ij,ij->i', a[valid], b[valid]) / (na[valid] * nb[valid]), -1, 1)
        ratio[sl][valid] = np.minimum(na[valid], nb[valid]) / np.maximum(na[valid], nb[valid])
        error[sl][nb > 0] = np.linalg.norm(a[nb > 0] - b[nb > 0], axis=1) / nb[nb > 0]
    return {'cosine': cosine, 'norm_ratio': ratio, 'replacement_relative_l2': error}


def distribution(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return {'n': len(v), 'mean': float(v.mean()), 'p10': float(np.quantile(v, .1)),
            'median': float(np.median(v)), 'p90': float(np.quantile(v, .9)),
            'min': float(v.min()), 'max': float(v.max())} if len(v) else {'n': 0}


def summarize(metrics, select=None):
    select = np.ones(len(metrics['cosine']), dtype=bool) if select is None else select
    c, r = metrics['cosine'][select], metrics['norm_ratio'][select]
    valid = np.isfinite(c)
    return {'sampled_pairs': int(select.sum()), 'valid_pairs': int(valid.sum()),
            'undefined_zero_norm_pairs': int((~valid).sum()),
            'cosine': distribution(c), 'norm_ratio': distribution(r),
            'replacement_relative_l2': distribution(metrics['replacement_relative_l2'][select]),
            'ge_0_99_pairs': int((c >= .99).sum()),
            'joint_0_99_pairs': int(((c >= .99) & (r >= .99)).sum())}


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for a, b in zip(starts, ends):
        result[order[a:b]] = (a + b - 1) / 2
    return result


def spearman(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    a, b = ranks(x), ranks(y)
    a -= a.mean(); b -= b.mean()
    return float(np.dot(a, b) / math.sqrt(float(np.dot(a, a) * np.dot(b, b))))


def conditional_trend(x, y, strata):
    groups = []
    order = np.argsort(strata, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(strata[order])) + 1, len(order)]
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            continue
        group = strata[order[start]]
        select = order[start:end]
        select = select[np.isfinite(y[select])]
        n = len(select)
        rho = spearman(x[select], y[select]) if n >= 50 else None
        groups.append({'stratum': int(group), 'valid_pairs': n, 'spearman': rho})
    measured = [g for g in groups if g['spearman'] is not None]
    total = sum(g['valid_pairs'] for g in measured)
    return {'groups': groups, 'measured_pairs': total,
            'weighted_spearman': sum(g['valid_pairs'] * g['spearman'] for g in measured) / total if total else None}


def trends(metrics, lengths, frequency):
    c = metrics['cosine']
    indicator = np.where(np.isfinite(c), (c >= .99).astype(np.float64), np.nan)
    return {name: {'frequency_within_length': conditional_trend(frequency, values, lengths),
                   'length_within_frequency_bin': conditional_trend(lengths, values, frequency_bins(frequency))}
            for name, values in (('cosine', c), ('ge_0_99', indicator))}


def covariate_trend(x, y, axes):
    groups, inverse = np.unique(np.column_stack(axes), axis=0, return_inverse=True)
    result = conditional_trend(x, y, inverse)
    result['stratum_values'] = groups.tolist()
    return result


def ancestry_diagnostics(metrics, keys, counts, pairs, signatures, child_controls, rows):
    a, b = pairs.T
    la = np.array([len(keys[i]) for i in a]); lb = np.array([len(keys[i]) for i in b])
    fa, fb = counts[a], counts[b]
    ba, bb = frequency_bins(fa), frequency_bins(fb)
    c = metrics['cosine']
    if (fb > fa).any():
        raise ValueError('ancestor occurrences must cover all descendant occurrences')
    fraction = fb / fa
    result = {'ancestor_frequency_holding_lengths_and_child_bin': covariate_trend(fa, c, (la, lb, bb)),
              'child_frequency_holding_lengths_and_ancestor_bin': covariate_trend(fb, c, (la, lb, ba)),
              'child_length_holding_ancestor_length_and_frequency_bins': covariate_trend(lb, c, (la, ba, bb)),
              'ancestor_length_holding_child_length_and_frequency_bins': covariate_trend(la, c, (lb, ba, bb)),
              'coverage_fraction_holding_lengths': covariate_trend(fraction, c, (la, lb)), 'coverage_cells': []}
    for lower, upper in ((0, .25), (.25, .5), (.5, .75), (.75, .9), (.9, 1), (1, 1)):
        select = (fraction == 1) if lower == upper else ((fraction >= lower) & (fraction < upper))
        result['coverage_cells'].append({'lower': lower, 'upper': upper,
                                         'source': summarize(metrics, select)})
    if signatures is not None:
        same = signatures[a] == signatures[b]
        if ((fa == fb) & ~same).any():
            raise ValueError('equal ancestor/child occurrence counts must imply equal host multiplicities')
        result['identical_incidence_frequency_holding_lengths'] = covariate_trend(
            fb, np.where(same, c, np.nan), (la, lb))
    if child_controls is not None:
        matched = child_controls >= 0
        comparison = pair_metrics(rows, np.column_stack((child_controls[matched], b[matched])))
        result['child_anchored_source'] = summarize(metrics, matched)
        result['child_anchored_control'] = summarize(comparison)
        result['child_anchored_unmatched_pairs'] = int((~matched).sum())
        ratios = np.maximum(fa[matched], counts[child_controls[matched]]) / np.minimum(fa[matched], counts[child_controls[matched]])
        result['child_anchored_frequency_ratio'] = distribution(ratios)
        source_best, control_best, joint_source, joint_control = [], [], [], []
        children = b[matched]
        order = np.argsort(children, kind='stable')
        boundaries = np.r_[0, np.flatnonzero(np.diff(children[order])) + 1, len(order)]
        source_cosines, source_ratios = c[matched], metrics['norm_ratio'][matched]
        for start, end in zip(boundaries, boundaries[1:]):
            if start == end:
                continue
            select = order[start:end]
            source_cos, source_ratio = source_cosines[select], source_ratios[select]
            control_cos, control_ratio = comparison['cosine'][select], comparison['norm_ratio'][select]
            if not np.isfinite(source_cos).all() or not np.isfinite(control_cos).all():
                continue
            source_best.append(float(source_cos.max())); control_best.append(float(control_cos.max()))
            joint_source.append(bool(((source_cos >= .99) & (source_ratio >= .99)).any()))
            joint_control.append(bool(((control_cos >= .99) & (control_ratio >= .99)).any()))
        result['best_per_child_matched_candidate_count'] = {
            'candidate_rows': len(source_best), 'source_best_cosine': distribution(source_best),
            'control_best_cosine': distribution(control_best),
            'source_ge_0_99_rows': sum(v >= .99 for v in source_best),
            'control_ge_0_99_rows': sum(v >= .99 for v in control_best),
            'source_joint_0_99_rows': sum(joint_source), 'control_joint_0_99_rows': sum(joint_control),
            'scope': 'Equal matched candidate counts per descendant; this is not direct-retained compaction'}
    return result


def audit_relation(rows, keys, counts, relation, *, initial=None, signatures=None, exposures=None, limit=LIMIT, prepared=None):
    if prepared is None:
        pairs, population = structural_pairs(keys, relation, limit)
        controls = matched_controls(keys, counts, pairs)
    else:
        pairs, population, controls = prepared[:3]
    source = pair_metrics(rows, pairs)
    matches = controls >= 0
    comparison = pair_metrics(rows, np.column_stack((pairs[matches, 0], controls[matches])))
    length = np.asarray([len(keys[b]) for _, b in pairs])
    frequency = counts[pairs[:, 1]] if relation in ('parent', 'ancestor') else counts[pairs].max(axis=1)
    bins = frequency_bins(frequency)
    cells = []
    for l, f in sorted(set(zip(length.tolist(), bins.tolist()))):
        select = (length == l) & (bins == f)
        cells.append({'length': l, 'frequency_bin': f, 'structural': summarize(source, select),
                      'matched_control': summarize(comparison, select[matches])})
    source_matched = summarize(source, matches)
    control_summary = summarize(comparison)
    a, b = pairs[matches].T
    valid_comparison = np.isfinite(source['cosine'][matches]) & np.isfinite(comparison['cosine'])
    ratios = np.maximum(counts[b], counts[controls[matches]]) / np.minimum(counts[b], counts[controls[matches]])
    result = {'relation': relation, 'population_pairs': population, 'sampled_pairs': len(pairs),
              'sample_is_complete': population == len(pairs), 'sampling_seed': SEED,
              'source': summarize(source), 'unmatched_pairs': int((~matches).sum()),
              'matched_source': source_matched, 'matched_control': control_summary,
              'matched_frequency_exact_fraction': float((counts[b] == counts[controls[matches]]).mean()) if len(b) else None,
              'matched_frequency_ratio': distribution(ratios),
              'paired_mean_cosine_difference': float((source['cosine'][matches][valid_comparison] - comparison['cosine'][valid_comparison]).mean()) if valid_comparison.any() else None,
              'cells': cells, 'conditional_trends': trends(source, length, frequency),
              'max_frequency_trends': trends(source, length, counts[pairs].max(axis=1)),
              'child_parent_frequency_ratio': distribution(counts[pairs[:, 1]] / counts[pairs[:, 0]]) if relation in ('parent', 'ancestor') else None}
    for name, gate in [('cosine_0_99', source['cosine'] >= .99),
                       ('cosine_and_norm_0_99', (source['cosine'] >= .99) & (source['norm_ratio'] >= .99))]:
        touched = np.unique(pairs[gate].reshape(-1))
        result[name + '_observed_coverage'] = {'unique_rows_in_sample': len(touched),
            'total_source_rows': len(rows), 'occurrence_mass': int(counts[touched].sum()),
            'total_source_occurrence_mass': int(counts.sum()), 'full_population_coverage': population == len(pairs)}
    if signatures is not None:
        same = signatures[pairs[:, 0]] == signatures[pairs[:, 1]]
        result['identical_incidence'] = summarize(source, same)
        result['different_incidence'] = summarize(source, ~same)
        result['matched_control_identical_incidence'] = summarize(comparison, signatures[a] == signatures[controls[matches]])
    if relation in ('parent', 'ancestor'):
        child_controls = prepared[3] if prepared is not None and len(prepared) == 4 else None
        result['ancestry_diagnostics'] = ancestry_diagnostics(source, keys, counts, pairs, signatures, child_controls, rows)
    if initial is not None:
        initial_metrics = pair_metrics(initial, pairs)
        update_metrics = pair_metrics(rows.astype(np.float64) - initial.astype(np.float64), pairs)
        result['initial'] = summarize(initial_metrics)
        result['accumulated_update'] = summarize(update_metrics)
        result['excluding_initial_ge_0_99'] = summarize(source, initial_metrics['cosine'] < .99)
        result['excluding_initial_ge_0_99_trends'] = trends(
            {k: np.where(initial_metrics['cosine'] < .99, v, np.nan) for k, v in source.items()}, length, frequency)
    if exposures is not None:
        exposure = exposures[pairs[:, 1]] if relation in ('parent', 'ancestor') else exposures[pairs].max(axis=1)
        # Unupdated rows are disclosed, not mislabelled as frequency one.
        updated = exposure > 0
        result['zero_sampled_exposure_pairs'] = int((~updated).sum())
        result['sampled_exposure_trends'] = trends({k: v[updated] for k, v in source.items()}, length[updated], exposure[updated])
    return result
