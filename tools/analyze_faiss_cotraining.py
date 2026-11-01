"""Analyze same-task trained-row FAISS comparisons without choosing on test."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def block_interval(a, b, positions, repeats=1000, seed=42):
    groups, inverse = np.unique(positions // 128, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    totals = {prefix + key: np.bincount(inverse, weights=scores[key])
              for prefix, scores in (('a', a), ('b', b)) for key in ('loss', 'discrimination')}
    rng = np.random.default_rng(seed)
    intervals = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(groups), len(groups))
        intervals.append([totals['aloss'][chosen].sum() / totals['bloss'][chosen].sum() - 1,
                          (totals['adiscrimination'][chosen].sum() - totals['bdiscrimination'][chosen].sum()) / count[chosen].sum()])
    return {'token_block_size': 128, 'observed_blocks': len(groups), 'repeats': repeats, 'seed': seed,
            'loss_relative_change_95_percent_interval': np.quantile(intervals, [.025, .975], axis=0)[:, 0].tolist(),
            'discrimination_change_95_percent_interval': np.quantile(intervals, [.025, .975], axis=0)[:, 1].tolist(),
            'scope': 'Paired within-text token-block bootstrap; exploratory intervals, not independent corpora or simultaneous bounds.'}


def analyze(report, root=None):
    rows, selections = [], []
    for run in report['runs']:
        by_name = {v['name']: v for v in run['variants']}
        for v in run['variants']:
            codec = v['codec']['codec']
            family = 'hybrid' if v['name'].endswith('-hybrid') else 'group' if v['name'].startswith('group-') else 'direct'
            row = {'order': run['order'], 'seed': run['seed'], 'name': v['name'], 'family': family,
                   'codec': codec, 'model_bytes': v['model_bytes'],
                   'model_plus_fixed_plans_bytes': v['model_bytes'] + v['fixed_trace_plan_bytes'],
                   'known_aggregate_relative_l2': v['known_word_aggregate_fidelity']['global_relative_l2'],
                   'known_p50_ms': v['query']['known']['p50_ms'],
                   'known_dynamic_p50_ms': v['query']['known']['with_schedule_compilation']['p50_ms'],
                   'known_p95_ms': v['query']['known']['p95_ms'],
                   'oov_p50_ms': v['query']['oov']['p50_ms'],
                   'oov_accepted': v['query']['oov']['fidelity']['accepted'],
                   'oov_total': v['query']['oov']['fidelity']['total'],
                   'query_peak_rss_bytes': v['query_peak_rss_bytes'],
                   'training_seconds': v['codec']['training_seconds'],
                   'encoding_seconds': v['codec']['encoding_seconds'],
                   'validation': v['quality']['validation'], 'test': v['quality']['test']}
            direct = by_name.get('direct-' + codec)
            if direct and family != 'direct':
                row['versus_matched_direct'] = {
                    'model_saving_fraction': 1 - v['model_bytes'] / direct['model_bytes'],
                    'known_replay_speedup': direct['query']['known']['p50_ms'] / v['query']['known']['p50_ms'],
                    'known_dynamic_speedup': direct['query']['known']['with_schedule_compilation']['p50_ms'] / v['query']['known']['with_schedule_compilation']['p50_ms'],
                    'aggregate_error_ratio': row['known_aggregate_relative_l2'] / max(direct['known_word_aggregate_fidelity']['global_relative_l2'], 1e-30)}
            rows.append(row)
        for budget, selected in run['selection'].items():
            for family, item in selected.items():
                row = {'order': run['order'], 'seed': run['seed'], 'loss_budget': float(budget),
                       'family': family, 'selected': item}
                if item is not None and root is not None:
                    directory = root / f"order-{run['order']}-seed-{run['seed']}"
                    candidate = dict(np.load(directory / (item['name'] + '-test-scores.npz')))
                    baseline = dict(np.load(directory / 'common/test-scores.npz'))
                    positions = np.load(directory / 'common/test-evaluation.npz')['positions']
                    row['test_paired_block_interval'] = block_interval(candidate, baseline, positions)
                selections.append(row)
    return {'source_commit': report['source_commit'], 'status': report['status'],
            'runs': len(report['runs']), 'variants': len(rows),
            'failed_runs': [{ 'order': r['order'], 'seed': r['seed'], 'failures': r['failures']}
                            for r in report['runs'] if r['failures']],
            'rows': rows, 'validation_selected': selections,
            'interpretation': ['Known-word closed-domain comparison permits direct and certified-only groups on the SAME bags.',
                'OOV/all-domain comparisons must use direct or hybrid; rejected work is never counted as a speedup.',
                'Reconstruction error, finite SGNS quality and query cost are separate outcomes.',
                'Codebook/transform overhead is material on small tables; not extrapolated to million-row scale.',
                'Matched-codec grouping savings are distinct from choosing a different codec on validation.']}


def figures(analysis, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'direct': ('o', '#356bb3'), 'group': ('^', '#db7e1a'), 'hybrid': ('x', '#ad4663')}
    for metric, filename, ylabel in (
            ('known_aggregate_relative_l2', 'capacity-error.svg', 'Known-word aggregate relative L2'),
            ('known_p50_ms', 'capacity-replay.svg', '512-bag replay p50 (ms)'),
            ('known_dynamic_p50_ms', 'capacity-dynamic.svg', 'ID schedule + query p50 (ms)')):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, order in zip(axes, (4, 8, 12)):
            for family, (marker, color) in styles.items():
                for codec in sorted({r['codec'] for r in analysis['rows']}):
                    values = [r for r in analysis['rows'] if r['order'] == order and r['family'] == family and r['codec'] == codec]
                    if not values:
                        continue
                    x = np.median([v['model_bytes'] / 2**20 for v in values])
                    y = max(1e-9, np.median([v[metric] for v in values]))
                    ax.scatter(x, y, marker=marker, color=color, s=45)
                    if family != 'hybrid':
                        ax.annotate(codec, (x, y), fontsize=6, xytext=(3, 3), textcoords='offset points', color=color)
                ax.scatter([], [], marker=marker, color=color, label=family)
            ax.set_title(f'Max n-gram order {order}')
            ax.set_xlabel('Complete model (MiB)')
            ax.set_yscale('log')
            ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.suptitle('Finite SGNS / known-word domain; medians over 3 training seeds', fontsize=11)
        fig.text(.5, .01, 'Word/output rows, registry, codebooks and maps counted; hybrid includes full original-row fallback. Zero error plotted at 1e-9.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .94))
        fig.savefig(output / filename)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--raw-root', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--plots', action='store_true')
    cfg = p.parse_args()
    raw = gzip.decompress(cfg.report.read_bytes()) if cfg.report.suffix == '.gz' else cfg.report.read_bytes()
    analysis = analyze(json.loads(raw), cfg.raw_root)
    analysis['report_sha256'] = hashlib.sha256(raw).hexdigest()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
    if cfg.plots:
        figures(analysis, cfg.output_dir)
    print(json.dumps({'runs': analysis['runs'], 'variants': analysis['variants'], 'failed_runs': analysis['failed_runs']}))


if __name__ == '__main__':
    main()
