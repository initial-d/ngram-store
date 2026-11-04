"""Validate/redraw the public residual-bound reports without trained inputs."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median


BUDGETS = [0., .0001, .001, .005, .01, .02, .03, .05, .1, .2, .5, 1.]
EXPECTED = set(itertools.product(('legacy', 'average'), (4, 8, 12), (3, 7, 11)))


def read_report_bytes(path):
    path = Path(path)
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == '.gz' else raw


def validate(report):
    if report.get('status') != 'passed' or report.get('failures') or report.get('budgets') != BUDGETS:
        raise ValueError('complete passed predeclared report required')
    keys = [(r['convention'], r['order'], r['seed']) for r in report['runs']]
    if len(keys) != 18 or set(keys) != EXPECTED:
        raise ValueError('all 18 unique frozen checkpoints required')
    for run in report['runs']:
        if not run['canonical_rows_identical'] or run['capacity']['source_fallback_retained']:
            raise ValueError('canonical/fallback contract changed')
        c = run['capacity']
        if (c['raw_complete_model_bytes'] != c['raw_ngram_fp32_bytes'] + c['unchanged_word_output_registry_bytes']
                or c['budgeted_complete_model_bytes'] != c['native_certified_bytes'] + c['sidecar_bytes'] + c['unchanged_word_output_registry_bytes']):
            raise ValueError('complete-model byte accounting mismatch')
        if c['complete_model_saving_vs_raw_fp32'] != 1 - c['budgeted_complete_model_bytes'] / c['raw_complete_model_bytes']:
            raise ValueError('saving denominator mismatch')
        if c['sidecar_bytes'] < 8 * run['partition']['original_rows'] + 16 * run['partition']['group_rows']:
            raise ValueError('residual arrays omitted from capacity')
        for kind in ('known', 'oov'):
            trace = run['traces'][kind]
            if not trace['all_absolute_and_relative_bounds_pass'] or not trace['strict_native_output_bytes_identical']:
                raise ValueError('numerical/strict-output failure')
            arrays = [trace[name] for name in ('absolute_errors', 'relative_errors', 'absolute_l2_bounds',
                       'relative_l2_bounds', 'nonempty_mask', 'strict_mask')]
            if any(len(a) != trace['queries'] for a in arrays):
                raise ValueError('one assessment per query required')
            error, relative, bound, relbound, nonempty, strict = arrays
            if any(not math.isfinite(v) or v < 0 for a in (error, bound) for v in a):
                raise ValueError('invalid absolute-error observation')
            if any(v is not None and (not math.isfinite(v) or v < 0) for a in (relative, relbound) for v in a):
                raise ValueError('invalid relative-error observation')
            if any(e > b for e, b in zip(error, bound)) or any(b is not None and (e is None or e > b)
                    for e, b in zip(relative, relbound)):
                raise ValueError('actual error exceeds published bound')
            if trace['nonempty'] != sum(nonempty) or trace['empty'] != len(nonempty) - sum(nonempty):
                raise ValueError('empty-query accounting mismatch')
            count = sum(s and n for s, n in zip(strict, nonempty))
            if trace['strict_nonempty'] != count or trace['strict_coverage'] != count / max(sum(nonempty), 1):
                raise ValueError('strict coverage mismatch')
            if [p['budget'] for p in trace['points']] != BUDGETS:
                raise ValueError('predeclared budgets changed')
            for point in trace['points']:
                budget = point['budget']
                accepted = [n and b is not None and b <= budget for n, b in zip(nonempty, relbound)]
                count = sum(accepted)
                if point['accepted_nonempty'] != count or point['coverage'] != count / max(sum(nonempty), 1):
                    raise ValueError('budget coverage mismatch')
                if point['new_nonempty_vs_strict'] != sum(a and not s for a, s in zip(accepted, strict)):
                    raise ValueError('new-query count mismatch')
                if point['oracle_eligible_nonempty'] != sum(n and e is not None and e <= budget for n, e in zip(nonempty, relative)):
                    raise ValueError('oracle-eligibility mismatch')
                maximum = max((e for a, e in zip(accepted, relative) if a), default=None)
                if point['maximum_actual_relative_l2'] != maximum or (maximum is not None and maximum > budget):
                    raise ValueError('admitted-query error mismatch')
    return report


def summarize(report, retiming=None):
    validate(report)
    summary = {'source_commit': report['source_commit'], 'conventions': {},
        'all_18_checkpoint_absolute_and_relative_checks_pass': True,
        'budgets': BUDGETS, 'scope': 'Existing finite SGNS weights; per-query n-gram aggregate L2, not semantic or LLM quality.'}
    if retiming is not None:
        keys = [(r['convention'], r['order'], r['seed']) for r in retiming['runs']]
        if retiming.get('status') != 'passed' or len(keys) != 18 or set(keys) != EXPECTED:
            raise ValueError('complete unique retiming/equality controls required')
        for run in retiming['runs']:
            original = next(r for r in report['runs'] if (r['convention'], r['order'], r['seed']) ==
                            (run['convention'], run['order'], run['seed']))
            for kind, trace in run['traces'].items():
                if not trace['all_output_bound_domain_bytes_identical'] or trace['trace_sha256'] != original['trace_sha256'][kind]:
                    raise ValueError('retiming changed numerical contract or queries')
                if run['convention'] == 'legacy':
                    if len(trace['reference_seconds']) != 5 or len(trace['batched_seconds']) != 5 or len(trace['order']) != 5:
                        raise ValueError('five paired timing samples required')
                    if any(not math.isfinite(v) or v <= 0 for name in ('reference_seconds', 'batched_seconds') for v in trace[name]):
                        raise ValueError('invalid timing observation')
                    if (trace['reference_p50_seconds'] != median(trace['reference_seconds'])
                            or trace['batched_p50_seconds'] != median(trace['batched_seconds'])
                            or trace['improvement'] != trace['reference_p50_seconds'] / trace['batched_p50_seconds']):
                        raise ValueError('timing summary mismatch')
        summary['retiming_source_commit'] = retiming['source_commit']
        summary['all_18_batched_output_bound_domain_bytes_identical'] = True
    for convention in ('legacy', 'average'):
        rows = []
        for order in (4, 8, 12):
            selected = [r for r in report['runs'] if r['convention'] == convention and r['order'] == order]
            row = {'order': order, 'strict_oov_coverage': median(r['traces']['oov']['strict_coverage'] for r in selected),
                'oov_budget_coverage': [median(r['traces']['oov']['points'][i]['coverage'] for r in selected) for i in range(len(BUDGETS))],
                'maximum_accepted_relative_l2': [max((r['traces']['oov']['points'][i]['maximum_actual_relative_l2']
                    for r in selected if r['traces']['oov']['points'][i]['maximum_actual_relative_l2'] is not None), default=None)
                    for i in range(len(BUDGETS))],
                'raw_complete_model_bytes': median(r['capacity']['raw_complete_model_bytes'] for r in selected),
                'budgeted_complete_model_bytes': median(r['capacity']['budgeted_complete_model_bytes'] for r in selected),
                'sidecar_bytes': median(r['capacity']['sidecar_bytes'] for r in selected),
                'complete_model_saving_vs_raw_fp32': median(r['capacity']['complete_model_saving_vs_raw_fp32'] for r in selected),
                'initial_full_oov_assess_ms': median(r['traces']['oov']['cost']['full_assess_all_queries']['p50_seconds'] * 1000 for r in selected),
                'source_fp64_oov_ms': median(r['traces']['oov']['cost']['source_fp64_all_queries']['p50_seconds'] * 1000 for r in selected)}
            if retiming is not None and convention == 'legacy':
                matching = [r['traces']['oov'] for r in retiming['runs'] if r['convention'] == convention and r['order'] == order]
                row.update({'paired_reference_oov_ms': median(r['reference_p50_seconds'] * 1000 for r in matching),
                    'batched_full_oov_assess_ms': median(r['batched_p50_seconds'] * 1000 for r in matching),
                    'paired_oov_improvement': median(r['improvement'] for r in matching)})
            rows.append(row)
        summary['conventions'][convention] = rows
    return summary


def plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    colors = ['#2458a6', '#bd6412', '#1d856e']
    for ax, convention in zip(axes, ('legacy', 'average')):
        for row, color in zip(summary['conventions'][convention], colors):
            ax.semilogx([b * 100 for b in BUDGETS[1:]], [v * 100 for v in row['oov_budget_coverage'][1:]],
                        color=color, marker='o', markersize=3, linewidth=1.6, label=f"order {row['order']}")
        for index, label in ((0, 'strict order 4'), (1, 'strict order 8/12')):
            coverage = summary['conventions'][convention][index]['strict_oov_coverage'] * 100
            ax.axhline(coverage, color=colors[index], linestyle='--', linewidth=.9, label=label)
        ax.set_title('Legacy SGD (primary)' if convention == 'legacy' else 'Average derivative (sensitivity)', fontsize=11)
        ax.set_xlabel('Allowed per-query relative L2 upper bound (%)')
        ax.set_ylim(0, 105)
        ax.set_xticks([.01, .1, 1, 5, 10, 100], labels=['0.01', '0.1', '1', '5', '10', '100'])
        ax.grid(True, alpha=.2)
    axes[0].set_ylabel('Nonempty stored-ngram OOV queries admitted (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=8)
    fig.suptitle('Source-specific residual bounds expand query coverage', fontsize=12)
    fig.text(.5, .085, 'enwik8 prefix · frozen finite SGNS · 512 OOV queries · three-seed medians · not downstream quality',
             ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=[0, .11, 1, .95])
    for extension in ('svg', 'png'):
        fig.savefig(output / ('oov-error-budget.' + extension), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--retiming', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--plots', action='store_true')
    cfg = p.parse_args()
    raw = read_report_bytes(cfg.report)
    report = json.loads(raw)
    retiming = json.loads(cfg.retiming.read_text()) if cfg.retiming else None
    if retiming is not None and retiming['frozen_report_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('retiming references a different frozen report')
    summary = summarize(report, retiming)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if cfg.plots:
        plots(summary, cfg.output_dir)
    print(json.dumps({'status': 'passed', 'checkpoints': len(report['runs']),
        'output': str(cfg.output_dir / 'summary.json')}, indent=2))


if __name__ == '__main__':
    main()
