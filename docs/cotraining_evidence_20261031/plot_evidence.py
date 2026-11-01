"""Scientific figures from measured JSON only; no fabricated vectors/curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180})


def read(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    fig.subplots_adjust(left=.10, right=.98, bottom=.28, top=.83, wspace=.38)
    fig.savefig(ROOT / (name + '.png'), bbox_inches='tight')
    svg = ROOT / (name + '.svg')
    fig.savefig(svg, bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True)
    colors = ['#2978b5', '#d88420', '#369363']
    for ax, study, title in zip(axes, ['legacy-report.json', 'average-report.json'],
                              ['FastText-style unscaled input gradient', 'Standard average-bag derivative']):
        runs = read(study)['runs']
        for order, color in zip([4, 8, 12], colors):
            selected = [r for r in runs if r['order'] == order]
            steps = [x['step'] for x in selected[0]['dynamics']['snapshots']]
            values = np.asarray([[x['identical_incidence_pairs']['cosine']['mean']
                                   for x in r['dynamics']['snapshots']] for r in selected])
            ax.plot(steps, values.mean(0), marker='o', color=color, label=f'Max order {order}')
            ax.fill_between(steps, values.min(0), values.max(0), color=color, alpha=.12)
        ax.axhline(.99, color='#888888', linestyle=':', linewidth=1)
        ax.set(title=title, xlabel='SGD steps', ylim=(-.03, 1.03))
        ax.legend(loc='lower right', frameon=False)
    axes[0].set_ylabel('Mean final-vector cosine\nwithin identical incidence groups')
    fig.suptitle('Shared updates align; final cosine depends on the training rule', y=.97)
    fig.text(.02, .03, 'Actual independent initialization; 3 seeds per order; finite CPU SGNS. Shading: seed min/max.\n'
                       'At step 4000 the measured update cosine is >0.999999999999 for both rules; not LLM/convergence evidence.', fontsize=9)
    save(fig, 'training-dynamics')

    cached = read('cached-report.json')['runs']
    audit = read('final-audit.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    labels = ['Source native', 'Joint 0.99', 'Canonical + DATrie + byte dedup', 'Precomputed words']
    variant_names = ['identity', 'original-joint099', 'canonical-datrie-exact']
    x = np.arange(3)
    width = .19
    for j, label in enumerate(labels):
        totals = []
        for order in [4, 8, 12]:
            if j < 3:
                values = [next(v['complete_input_output_key_bytes'] for v in r['variants'] if v['name'] == variant_names[j])
                          for r in cached if r['order'] == order]
            else:
                values = [r['precomputed']['complete_encoded_word_output_key_bytes'] for r in audit['runs']
                          if r['study'] == 'cached-results' and r['order'] == order]
            totals.append(np.mean(values) / 2**20)
        axes[0].bar(x + (j - 1.5) * width, totals, width, label=label,
                    color=['#8795a5', '#d88420', '#2978b5', '#369363'][j])
    axes[0].set(xticks=x, xticklabels=['4', '8', '12'], xlabel='Maximum n-gram order',
                ylabel='Input + output + keys/certificates (MiB)', ylim=(0, 10.5),
                title='Nine cached SGNS references')
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    large, raw = read('large-report.json'), read('raw-block-evidence.json')
    labels = ['Source\nnative', 'Certified\nencoded', 'Certified\nraw blocks', 'Precomputed\nknown words']
    values = [large['variants'][0]['complete_subset_bytes'], large['complete_certified_subset_and_plan_bytes'],
              raw['complete_subset_and_plan_bytes'], audit['large_precomputed']['complete_encoded_word_table_and_key_bytes']]
    axes[1].bar(np.arange(4), np.asarray(values)/2**20, color=['#8795a5', '#2978b5', '#d88420', '#369363'])
    axes[1].set(xticks=np.arange(4), xticklabels=labels, ylabel='Subset + keys/maps/certificate bytes (MiB)',
                title='1.86M actual updated rows × 256 FP32')
    fig.text(.02, .03, 'Certified/canonical variants change source rows; only certified aggregates are equivalent in real arithmetic.\n'
                       'Precomputed words have a narrower inference-only contract and win within that scope. Large word/output model unavailable.\n'
                       'Small compiled plans are reported separately; large certified bars include the measured 512-bag plan.', fontsize=9)
    save(fig, 'capacity-and-strong-baselines')

    samples = raw['query_samples_seconds']
    keys = ['encoded-certified', 'raw-certified', 'raw-mmap', 'precomputed-known-words']
    values = [np.median(samples[k][1:]) * 1000 for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    axes[0].bar(range(4), values, color=['#2978b5', '#d88420', '#8795a5', '#369363'])
    axes[0].set(xticks=range(4), xticklabels=['Encoded\ncertified', 'Raw-block\ncertified', 'Raw\nmmap', 'Precomputed\nwords'],
                yscale='log', ylabel='Median repeated batch query (ms, log scale)', title='Same 512 known-word query bags')
    for i, value in enumerate(values):
        axes[0].text(i, value * 1.18, f'{value:.2f}', ha='center', fontsize=9)
    oov = [next(r for r in cached if r['order'] == order)['oov']['test'] for order in [4, 8, 12]]
    rates = [r['rejected_oov_types'] / r['audited_oov_types'] * 100 for r in oov]
    rates.append(large['oov']['rejected'] / large['oov']['audited_types'] * 100)
    axes[1].bar(range(4), rates, color='#bd6262')
    axes[1].set(xticks=range(4), xticklabels=['Small 4', 'Small 8', 'Small 12', 'Large 12'],
                ylabel='Unseen word types rejected by certificate (%)', ylim=(0, 100), title='Scope limits remain material')
    fig.text(.02, .03, 'Warm OS pages, shared CPU, 8 repeated timings; preprocessing excluded. Native still trails raw mmap.\n'
                       'Small OOV: all 2486 test types; large: 4096 seeded sampled types. Rejection is a scope test, not a semantic quality score.', fontsize=9)
    save(fig, 'query-cost-and-domain')

    isolation = read('planner-isolation-report.json')['runs']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    orders = [4, 8, 12]
    for j, (field, label, color) in enumerate([
        ('datrie_extra_saving_over_same_canonical_codec', 'DATrie only (no byte dedup)', '#2978b5'),
        ('byte_dedup_extra_saving_over_same_canonical_codec', 'Byte dedup only (no DATrie)', '#369363'),
    ]):
        values = np.asarray([[100 * r[field] for r in isolation if r['order'] == order] for order in orders])
        axes[0].bar(np.arange(3) + (j - .5) * .34, values.mean(1), .34, label=label, color=color)
    axes[0].set(xticks=np.arange(3), xticklabels=orders, xlabel='Maximum n-gram order',
                ylabel='Extra complete saving (%)\nover canonical codec-only', ylim=(0, 42),
                title='Same canonical rows\nIndependent sharing controls')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    small = read('layout-small-report.json')['runs']
    exact = [100 * np.mean([r['extra_complete_saving'] for r in small if r['source']['order'] == order])
             for order in orders]
    exact.append(100 * read('layout-large-report.json')['runs'][0]['extra_complete_saving'])
    axes[1].bar(range(4), exact, color='#8795a5')
    axes[1].set(xticks=range(4), xticklabels=['Small 4', 'Small 8', 'Small 12', 'Large 12'],
                ylabel='Extra complete saving (%)\nover original-row codec', ylim=(0, 2.2),
                title='Original source bytes\nIncidence reorder only')
    for i, value in enumerate(exact):
        axes[1].text(i, value + .06, f'{value:.3f}%', ha='center', fontsize=9)
    fig.text(.02, .03, 'Left: projection changes source vectors; every canonical byte/ID verified. Keys, outputs and certificates counted equally.\n'
                       'Right: every original source byte/ID preserved. Different denominator from left; savings cannot be added.\n'
                       'Three seeds for each small order; one actual 1.86M-row large reference. DATrie key-index size/latency not evaluated here.', fontsize=9)
    save(fig, 'planner-isolation')


if __name__ == '__main__':
    main()
