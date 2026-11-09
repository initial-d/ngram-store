"""Scientific figures from completed frozen numeric application evidence."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .tree_node_quality import load

COLORS = {'raw': '#777777', 'dense': '#999999', 'constant': '#aaaaaa', 'native-identity': '#1565c0',
          'tree-datrie': '#00897b', 'tree-reference': '#6a1b9a', 'faiss': '#e65100', 'faiss-dedup': '#ef9a00',
          'faiss-native': '#c62828', 'faiss-native-layout': '#ad1457', 'faiss-dedup-native': '#4e342e'}


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=220, bbox_inches='tight')
    fig.savefig(output / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def llm_capacity(root, output):
    summaries = load(root / 'analysis/llm-summary.json'); paired = load(root / 'analysis/llm-paired-comparisons.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, model in zip(axes, ('qwen06', 'qwen35')):
        groups = defaultdict(list)
        for item in summaries:
            if not item['model'].startswith(model):
                continue
            for choice in item['choices']:
                if choice['name']:
                    groups[(choice['family'], choice['budget_pp'])].append(choice)
        families = sorted({key[0] for key in groups})
        for family in families:
            x = []; y = []
            for budget in (.1, .5, 1., 2.):
                points = groups[(family, budget)]
                if len(points) != 3:
                    continue
                result = next(r for r in paired if r['model'] == model and r['choice'] == [budget, family])
                x.append(np.mean([p['complete_serving_bytes'] for p in points]) / (1 << 20))
                y.append(-result['observed_candidate_minus_source'][0] * 100)
            if x:
                ax.plot(x, y, 'o-', markersize=4, label=family, color=COLORS[family], linewidth=1.3)
        ax.axhline(.5, color='#444444', linestyle='--', linewidth=.8, label='Primary 0.5 pp budget')
        ax.axhline(0., color='#cccccc', linewidth=.7)
        ax.set(xlabel='Complete memory branch + required registry (MiB)', ylabel='Test harmonic-score decline (percentage points)',
               title='Qwen3-0.6B' if model == 'qwen06' else 'Qwen3.5-4B')
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, .5))
    fig.suptitle('All three seeds; calibration-selected methods audited on official test cases', fontsize=11)
    save(fig, output, 'llm-capacity-quality')


def independent_controls(root, output):
    paired = load(root / 'analysis/llm-paired-comparisons.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    labels = ('head-constant', 'adapter', 'hash')
    for ax, model in zip(axes, ('qwen06', 'qwen35')):
        rows = [next(r for r in paired if r['model'] == model and r['choice'] == ['independent', name]) for name in labels]
        observed = np.array([r['observed_candidate_minus_source'][0] for r in rows]) * 100
        intervals = np.array([r['paired_95_percent_intervals'][0] for r in rows]) * 100
        ax.bar(np.arange(3), observed, color=['#757575', '#6a1b9a', '#e65100'], alpha=.85)
        ax.vlines(np.arange(3), intervals[:, 0], intervals[:, 1], color='#222222', linewidth=1.4)
        ax.axhline(0., color='#444444', linewidth=.8)
        ax.set(xticks=np.arange(3), xticklabels=['Head constants', 'Matched adapter', 'Native OOV hash'],
               ylabel='Candidate minus full-table harmonic score (pp)',
               title='Qwen3-0.6B' if model == 'qwen06' else 'Qwen3.5-4B')
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Independent training controls; paired three-seed/case 95% intervals', fontsize=11)
    save(fig, output, 'independent-memory-controls')


def cpu_serving(root, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, model in zip(axes, ('qwen06', 'qwen35')):
        groups = defaultdict(list)
        for folder in sorted((root / 'llm/models').glob(model + '*')):
            grid = {e['build']['name']: e for e in load(folder / 'grid.json.gz')}
            for item in load(folder / 'retimings-cpu.json.gz'):
                r = item['request']; e = grid[r['name']]
                if r['cache']:
                    continue
                report = item['report']
                groups[e['build']['family']].append((e['complete_serving_bytes'] / (1 << 20), report['query']['test']['p50_ms']))
        for family, points in sorted(groups.items()):
            x, y = np.array(points).T
            ax.scatter(x, y, s=22, alpha=.7, label=family, color=COLORS[family])
        ax.set(xscale='log', yscale='log', xlabel='Complete memory serving bytes (MiB)',
               ylabel='Original tokens → final vectors, test request p50 (ms)',
               title='Qwen3-0.6B' if model == 'qwen06' else 'Qwen3.5-4B')
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, .5))
    fig.suptitle('Fresh serial CPU workers; every required lookup and codec decode included', fontsize=11)
    save(fig, output, 'cpu-capacity-query')


def training_points(root, output):
    training = load(root / 'long/training-selection.json')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, convention in enumerate(('legacy', 'average')):
        for col, order in enumerate((4, 8, 12)):
            ax = axes[row, col]
            for scale, color in ((1., '#1565c0'), (.1, '#00897b')):
                means = []; lows = []; highs = []
                for step in (2000, 8000, 32000):
                    values = [p['absolute_validation_loss'] for g in training['groups']
                              if g['convention'] == convention and g['order'] == order
                              for p in g['points'] if p['steps'] == step and p['scale'] == scale]
                    assert len(values) == 3
                    means.append(np.mean(values)); lows.append(min(values)); highs.append(max(values))
                ax.plot((2000, 8000, 32000), means, 'o-', color=color, label='Initialization scale ' + str(scale))
                ax.fill_between((2000, 8000, 32000), lows, highs, color=color, alpha=.13)
            ax.set(xscale='log', xlabel='Recorded training steps', ylabel='Absolute validation SGNS loss',
                   title=f'{convention}, order {order}')
            ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('All 108 trained checkpoints; shading is three-seed range, not a confidence interval', fontsize=11)
    save(fig, output, 'training-duration-initialization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    for function in (llm_capacity, independent_controls, cpu_serving, training_points):
        function(args.evidence, args.output)


if __name__ == '__main__':
    main()
