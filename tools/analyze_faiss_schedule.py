"""Summarize paired frozen-codec timings; no model selection or extrapolation."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def family(name):
    return 'hybrid' if name.endswith('-hybrid') else 'group' if name.startswith('group-') else 'direct'


def summarize(report):
    if report['status'] != 'completed' or report['failures']:
        raise ValueError('completed, failure-free retiming required')
    keys = {(r['order'], r['seed'], r['name']) for r in report['runs']}
    if len(keys) != 27 or len(report['runs']) != 27:
        raise ValueError('27 unique frozen selections required')
    rows = []
    for order in (4, 8, 12):
        for category in ('direct', 'group', 'hybrid'):
            runs = [r for r in report['runs'] if r['order'] == order and family(r['name']) == category]
            if {r['seed'] for r in runs} != {3, 7, 11}:
                raise ValueError('matched three-seed coverage required')
            for trace in ('known', 'oov'):
                if not all(r['all_registered_word_outputs_byte_identical'] and
                           r['query'][trace]['schedule_arrays_byte_identical'] and
                           r['query'][trace]['output_bytes_identical'] for r in runs):
                    raise ValueError('schedule/output equivalence failed')
                timings = [r['query'][trace]['timing'] for r in runs]
                rows.append({'order': order, 'family': category, 'trace': trace,
                    'median_model_bytes': statistics.median(r['model_bytes'] for r in runs),
                    'accepted_bags': sorted({r['query'][trace]['fidelity']['accepted'] for r in runs}),
                    'total_bags': sorted({r['query'][trace]['fidelity']['total'] for r in runs}),
                    'p50_ms': {key: statistics.median(t[key]['p50_ms'] for t in timings) for key in timings[0]},
                    'median_paired_dynamic_speedup': statistics.median(t['dynamic_old']['p50_ms'] /
                                                                     t['dynamic_new']['p50_ms'] for t in timings)})
    return {'benchmark_source_commit': report['source_commit'], 'variants': 27,
            'registered_output_checks': sum(r['registered_words'] for r in report['runs']),
            'rows': rows, 'scope': 'Three-seed medians; within-process paired timings; OOV includes rejected work.'}


def figure(summary, output, old_label='Scalar certificate', new_label='Batch certificate'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, order in zip(axes, (4, 8, 12)):
        rows = [next(r for r in summary['rows'] if r['order'] == order and
                     r['family'] == category and r['trace'] == 'known') for category in ('direct', 'group', 'hybrid')]
        for offset, key, color, label in ((-.18, 'dynamic_old', '#356bb3', old_label),
                                         (.18, 'dynamic_new', '#db7e1a', new_label)):
            ax.bar([i + offset for i in range(3)], [r['p50_ms'][key] for r in rows],
                   width=.36, color=color, label=label)
        ax.set_xticks(range(3), ['Direct', 'Certified', 'Hybrid'])
        ax.set_title(f'Max order {order}')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Compilation + query p50 (ms)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(.5, .95))
    fig.suptitle('512 known bags, frozen SQ6 codecs; three-seed medians')
    fig.text(.5, .015, 'Shared warm CPU, one query thread. Certified domain and full hybrid fallback costs are unchanged.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .84))
    svg = output / 'dynamic-query.svg'
    fig.savefig(svg)
    # Matplotlib writes trailing spaces in path attributes; normalize the
    # generated text without changing coordinates or rendering.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--plots', action='store_true')
    p.add_argument('--old-label', default='Scalar certificate')
    p.add_argument('--new-label', default='Batch certificate')
    cfg = p.parse_args()
    raw = cfg.report.read_bytes()
    summary = summarize(json.loads(raw))
    summary['report_sha256'] = hashlib.sha256(raw).hexdigest()
    summary['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if cfg.plots:
        figure(summary, cfg.output_dir, cfg.old_label, cfg.new_label)


if __name__ == '__main__':
    main()
