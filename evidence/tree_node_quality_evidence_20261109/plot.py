"""Standalone research figures from archived summaries; requires matplotlib."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from verify import load, gate

ROOT = Path(__file__).resolve().parent
COLORS = {'raw': '#737373', 'native-identity': '#b07aa1', 'tree-datrie': '#0072b2',
          'tree-reference': '#009e73', 'faiss': '#d55e00', 'faiss-dedup': '#e69f00', 'constant': '#cc79a7'}
LABELS = {'raw': 'Raw precision', 'native-identity': 'Native identity', 'tree-datrie': 'DATrie joint',
          'tree-reference': 'Independent tree reference', 'faiss': 'FAISS', 'faiss-dedup': 'FAISS + byte dedup', 'constant': 'Constant row'}


def legend(fig, axes, columns=4):
    entries = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        entries.update(zip(labels, handles))
    fig.legend(list(entries.values()), list(entries), loc='outside lower center', ncol=columns)


def save(fig, name):
    out = ROOT / 'figures'; out.mkdir(exist_ok=True)
    fig.savefig(out / (name + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(out / (name + '.svg'), bbox_inches='tight')
    svg = out / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    data = json.loads((ROOT / 'summary.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    selection = load(ROOT / 'selection.json')
    grids = {m['name']: load(ROOT / 'models' / m['name'] / 'grid.json.gz') for m in selection['models']}
    thresholds = (.99, .98, .97, .95, .90)
    for planner in ('datrie', 'reference'):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row, convention in enumerate(('legacy', 'average')):
            for col, order in enumerate((4, 8, 12)):
                ax = axes[row, col]; matrix, fails = [], []
                for cosine in thresholds:
                    values, invalid = [], []
                    for norm in thresholds:
                        records, ratios = [], []
                        for seed in (3, 7, 11):
                            grid = grids[f'{convention}-order-{order}-seed-{seed}']
                            r = next(g for g in grid if g['build']['name'] == f'tree-{planner}-c{cosine}-n{norm}-int8')
                            raw = next(g for g in grid if g['build']['name'] == 'raw-fp32')
                            records.append(r); ratios.append(r['complete_serving_bytes'] / raw['complete_serving_bytes'])
                        values.append(statistics.median(ratios))
                        invalid.append(any(not gate(r['build']['validation']['0'], .001) for r in records))
                    matrix.append(values); fails.append(invalid)
                image = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
                ax.set_xticks(range(5), [str(t) for t in thresholds]); ax.set_yticks(range(5), [str(t) for t in thresholds])
                ax.set_xlabel('Original norm-ratio gate'); ax.set_title(f'{convention}, max order {order}')
                if col == 0:
                    ax.set_ylabel('Original cosine gate')
                for i in range(5):
                    for j in range(5):
                        ax.text(j, i, f'{matrix[i][j]:.2f}' + (' x' if fails[i][j] else ''),
                                ha='center', va='center', fontsize=9, color='white' if matrix[i][j] < .5 else 'black')
        fig.colorbar(image, ax=axes, shrink=.8, label='Complete bytes / raw FP32 (median of 3 seeds)')
        fig.suptitle(f'{planner}: full cosine x norm-ratio grid, followed by INT8 encoding\n'
                    'x marks a validation primary-budget failure in any seed; gates are on original rows')
        save(fig, 'joint-gates-' + planner)
    for cached in (0, 1):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row, convention in enumerate(('legacy', 'average')):
            for col, order in enumerate((4, 8, 12)):
                ax = axes[row, col]
                selected = [r for r in data['frozen_choices'] if r['eligible'] and r['cached'] == cached
                            and r['convention'] == convention and r['order'] == order]
                for family in COLORS:
                    budgets = sorted({r['budget'] for r in selected if r['family'] == family})
                    ratios = [statistics.median(r['complete_fraction_of_raw_fp32'] for r in selected
                              if r['family'] == family and r['budget'] == b) for b in budgets]
                    if budgets:
                        ax.plot([b * 100 for b in budgets], ratios, 'o-', color=COLORS[family], label=LABELS[family], ms=4)
                ax.set_xscale('log'); ax.set_ylim(bottom=0)
                ax.axhline(1, color='#cccccc', lw=.8)
                ax.set_title(f'{convention}, max order {order}')
                ax.set_xlabel('Validation loss-increase budget (%)')
                if col == 0:
                    ax.set_ylabel('Complete bytes / raw FP32')
                ax.grid(alpha=.2)
        legend(fig, axes)
        fig.suptitle('Validation-selected capacity (median of available seeds; ineligible cells retained in ledger)\n' +
            ('Known-word source aggregates precomputed; OOV still queried' if cached else 'Ordinary subword inference; all + OOV quality gates'))
        save(fig, 'capacity-cached' if cached else 'capacity-ordinary')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, convention in enumerate(('legacy', 'average')):
        for col, order in enumerate((4, 8, 12)):
            ax = axes[row, col]
            for variant, color in (('datrie', '#0072b2'), ('reference', '#009e73')):
                for scale, style in ((1., '-'), (.1, '--')):
                    group = [r for r in data['mechanism_groups'] if (r['convention'], r['order'], r['scale']) == (convention, order, scale)]
                    ax.plot([r['steps'] for r in group], [r['variants'][variant]['median_native_table_saving'] * 100 for r in group],
                            marker='o', linestyle=style, color=color, label=f'{variant}, init x{scale:g}')
            ax.set_xscale('log'); ax.set_xticks([2000, 8000, 32000], ['2k', '8k', '32k']); ax.xaxis.set_minor_locator(NullLocator()); ax.set_title(f'{convention}, max order {order}')
            ax.set_xlabel('Fixed SGD steps'); ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel('Native n-gram bundle saving (%)')
    legend(fig, axes)
    fig.suptitle('Joint .99 cosine / .99 norm-ratio compaction after actual training\nMedian of 3 seeds; FP32, 8192-byte blocks; excludes common model files')
    save(fig, 'mechanism-capacity')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, convention in enumerate(('legacy', 'average')):
        for order, color in ((4, '#0072b2'), (8, '#009e73'), (12, '#d55e00')):
            for scale, style in ((1., '-'), (.1, '--')):
                group = [r for r in data['mechanism_groups'] if (r['convention'], r['order'], r['scale']) == (convention, order, scale)]
                for row, key in ((0, 'median_update_to_initial_norm_ratio'), (1, 'median_source_confirmation_loss')):
                    axes[row, col].plot([r['steps'] for r in group], [r[key] for r in group], marker='o',
                                       linestyle=style, color=color, label=f'order {order}, init x{scale:g}')
        for row in (0, 1):
            ax = axes[row, col]; ax.set_xscale('log'); ax.set_xticks([2000, 8000, 32000], ['2k', '8k', '32k']); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlabel('Fixed SGD steps'); ax.grid(alpha=.2)
            ax.set_title(convention)
        axes[0, col].set_yscale('log')
    axes[0, 0].set_ylabel('Accumulated update / initial row norm')
    axes[1, 0].set_ylabel('Source confirmation SGNS loss / pair')
    legend(fig, axes, columns=3)
    fig.suptitle('Initialization control changes training, not just compression\nMedian of 3 seeds; same pre-generated context/negative schedule')
    save(fig, 'mechanism-update-quality')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    primary = [r for r in data['frozen_choices'] if r['eligible'] and r['budget'] == .001]
    for row, cached in enumerate((0, 1)):
        for col, trace in enumerate(('known', 'oov')):
            ax = axes[row, col]
            for family in COLORS:
                group = [r for r in primary if r['cached'] == cached and r['family'] == family]
                if group:
                    for passed, marker in ((True, 'o'), (False, 'x')):
                        points = [r for r in group if (r['test_budget_pass'] and r['confirmation_budget_pass']) == passed]
                        if points:
                            ax.scatter([r['query_ratio_to_raw_fp32'][trace] for r in points],
                                       [r['complete_fraction_of_raw_fp32'] for r in points],
                                       c=COLORS[family], label=LABELS[family] + ('' if passed else ' (audit fail)'),
                                       marker=marker, s=25, alpha=.7)
            ax.axvline(1, color='#999999', lw=.8); ax.axhline(1, color='#999999', lw=.8)
            ax.set_xlabel('Complete query p50 / matched raw FP32')
            if cached and trace == 'known':
                ax.set_xlim(.9, 1.12); ax.set_xticks([.9, .95, 1., 1.05, 1.1])
            else:
                ax.set_xscale('log')
            ax.set_ylim(bottom=0); ax.grid(alpha=.2)
            ax.set_title(f'{trace}, ' + ('known cache' if cached else 'ordinary'))
            if col == 0:
                ax.set_ylabel('Complete bytes / raw FP32')
    legend(fig, axes)
    fig.suptitle('Validation-selected capacity and query cost at primary 0.1% budget\n'
                 'All 18 models; x = original-test or confirmation failure; sequential workers; DATrie / zero row cache')
    save(fig, 'capacity-query')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, cached in enumerate((0, 1)):
        for col, convention in enumerate(('legacy', 'average')):
            ax = axes[row, col]
            for i, name in enumerate(('constant-zero', 'constant-mean', 'constant-weighted-mean')):
                group = [r for r in data['necessity_controls'] if r['convention'] == convention and r['cached'] == cached]
                for s, style in (('all', 'o-'), ('oov_nonempty', 's--')):
                    y = [statistics.median(r['quality']['confirmation'][s]['loss_relative_change'] * 100
                         for r in group if r['name'] == name and r['order'] == order) for order in (4, 8, 12)]
                    ax.plot((4, 8, 12), y, style, color=('#0072b2', '#009e73', '#d55e00')[i],
                            label=name.replace('constant-', '') + ', ' + s)
            ax.axhline(.1, color='#666666', linestyle=':', label='Primary 0.1% budget')
            ax.set_title(convention + (', known precomputation' if cached else ', ordinary'))
            ax.set_xlabel('Maximum n-gram order'); ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel('Confirmation SGNS loss increase (%)')
    legend(fig, axes)
    fig.suptitle('Frozen row-identity necessity controls, including full-domain known-word precomputation\n'
                 'Same original bag denominator and output weights; median of 3 seeds; no qualification implied')
    save(fig, 'necessity-controls')


if __name__ == '__main__':
    main()
