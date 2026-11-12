"""Plot archived aggregates; seed ranges are descriptive, not confidence intervals."""
import json
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SERIES = [('tinyshakespeare', 'legacy', 'TinyShakespeare / legacy', '#2367a5'),
          ('enwik8', 'legacy', 'enwik8 / legacy', '#d06122'),
          ('enwik8', 'average', 'enwik8 / average derivative', '#26855a')]


def main():
    summary = json.loads((ROOT / 'summary.json').read_text())
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'ngram-ancestry-20261109'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for dataset, convention, label, color in SERIES:
        values = []
        for seed in (3, 7, 11):
            d = json.loads((ROOT / dataset / f'{convention}-order-12-s{seed}' / 'analysis.json').read_text())
            cells = d['relations'][1]['ancestry_diagnostics']['coverage_cells']
            values.append([100 * c['source']['ge_0_99_pairs'] / c['source']['sampled_pairs'] for c in cells])
        median = [statistics.median(v[i] for v in values) for i in range(6)]
        axes[0].plot(range(6), median, marker='o', color=color, label=label)
        axes[0].fill_between(range(6), [min(v[i] for v in values) for i in range(6)],
                             [max(v[i] for v in values) for i in range(6)], color=color, alpha=.12)
        selected = [m for m in summary['seed_medians'] if (m['dataset'], m['convention']) == (dataset, convention)]
        for j, q in enumerate((.99, .95)):
            y = [100 * next(a['native_byte_saving_fraction'] for a in m['capacity'] if a['norm_ratio'] == q) for m in selected]
            axes[1].plot([4, 8, 12], y, color=color, marker='o' if j == 0 else 's',
                         linestyle='-' if j == 0 else '--', label=f'{label}, norm >= {q}')
    axes[0].set_xticks(range(6), ['[0,.25)', '[.25,.5)', '[.5,.75)', '[.75,.9)', '[.9,1)', '1'])
    axes[0].set_xlabel('Longer-fragment occurrences / shorter-fragment occurrences')
    axes[0].set_ylabel('Node-containment pairs with cosine >= 0.99 (%)')
    axes[0].set_title('Prefix/suffix tree paths, maximum order 12\nMedian and range across 3 seeds')
    axes[0].legend(fontsize=9, loc='upper left')
    axes[1].set_xticks([4, 8, 12]); axes[1].set_xlabel('Maximum character n-gram order')
    axes[1].set_ylabel('Native table byte saving vs identity table (%)')
    axes[1].set_title('Approximate prefix/suffix compaction, cosine >= 0.99\nN-gram table only; task quality not measured')
    axes[1].legend(fontsize=8, loc='upper left')
    for ax in axes:
        ax.set_ylim(-2, 100 if ax is axes[0] else 75)
        ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT / 'ancestry.png', dpi=150)
    fig.savefig(ROOT / 'ancestry.svg', metadata={'Date': None})
    svg = ROOT / 'ancestry.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    (ROOT / 'plot-environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
        'command': 'artifacts/conditional-similarity-20261109/plot-env/bin/python docs/conditional_similarity_evidence_20261109/plot.py',
        'inputs': 'summary.json and all order-12 character analysis.json; archived metrics only',
        'uncertainty': 'Three-seed range, not an inferential confidence interval'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
