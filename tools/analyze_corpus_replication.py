"""Source-linked replication summary; no selection, refitting or extrapolation."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ORDERS = (4, 8, 12)
SEEDS = (3, 7, 11)
FAMILIES = ('direct', 'group', 'hybrid')


def family(name):
    if name.endswith('-hybrid'):
        return 'hybrid'
    if name.startswith('group-'):
        return 'group'
    if name.startswith(('direct-', 'dedup-')):
        return 'direct'
    raise ValueError('unrecognized query family: ' + name)


def check_query(query):
    expected = {(order, seed, kind) for order in ORDERS for seed in SEEDS for kind in FAMILIES}
    actual = {(r['order'], r['seed'], family(r['name'])) for r in query['runs']}
    if query['status'] != 'completed' or query['failures'] or len(query['runs']) != 27 or actual != expected:
        raise ValueError('exact complete selected-codec retiming required')
    for r in query['runs']:
        if (float(r['selected_loss_budget']) != .001 or r['registered_words'] != 1200 or
                not r['all_registered_word_outputs_byte_identical'] or
                set(r['query']) != {'known', 'oov'} or
                not all(t['schedule_arrays_byte_identical'] and t['output_bytes_identical']
                        for t in r['query'].values())):
            raise ValueError('selected query domain/byte equivalence failed')
        if r['query']['known']['fidelity']['accepted'] != 512 or r['query']['known']['fidelity']['total'] != 512:
            raise ValueError('all 512 known bags must be accepted')


def query_summary(query):
    check_query(query)
    rows = []
    for order in ORDERS:
        times = {}
        for kind in FAMILIES:
            runs = [r for r in query['runs'] if r['order'] == order and family(r['name']) == kind]
            times[kind] = {mode: median(r['query']['known']['timing'][mode]['p50_ms'] for r in runs)
                           for mode in ('dynamic_old', 'dynamic_new')}
            times[kind]['median_paired_speedup'] = median(
                r['query']['known']['timing']['dynamic_old']['p50_ms'] /
                r['query']['known']['timing']['dynamic_new']['p50_ms'] for r in runs)
        rows.append({'order': order, 'known_total_query': times})
    return rows


def median(values):
    return float(statistics.median(values))


def summarize(training, codecs, query, native):
    if training['status'] != 'completed' or training['failures']:
        raise ValueError('complete paired training required')
    if len(training['runs']) != 18 or {(r['convention'], r['order'], r['seed']) for r in training['runs']} != {
            (convention, order, seed) for convention in ('legacy', 'average')
            for order in (4, 8, 12) for seed in (3, 7, 11)}:
        raise ValueError('all 18 predeclared training pairs required')
    if codecs['status'] != 'completed' or codecs['failed_runs'] or codecs['runs'] != 9:
        raise ValueError('complete nine-checkpoint codec comparison required')
    if (len(codecs['rows']) != codecs['variants'] or
            len({(r['order'], r['seed'], r['name']) for r in codecs['rows']}) != codecs['variants']):
        raise ValueError('unique complete codec rows required')
    check_query(query)
    expected_selections = {(order, seed, kind, budget) for order in ORDERS for seed in SEEDS
                           for kind in FAMILIES for budget in (.0001, .001, .005)}
    if (len(codecs['validation_selected']) != 81 or
            {(r['order'], r['seed'], r['family'], r['loss_budget']) for r in codecs['validation_selected']} != expected_selections):
        raise ValueError('exact validation selection grid required')
    if native['status'] != 'completed' or not native['plan_arrays_byte_identical'] or not native['native_outputs_byte_identical']:
        raise ValueError('native plan/output equivalence required')
    dynamics = []
    for convention in ('legacy', 'average'):
        for order in (4, 8, 12):
            runs = [r for r in training['runs'] if r['convention'] == convention and r['order'] == order]
            dynamics.append({'convention': convention, 'order': order,
                'mean_final_pair_cosine': statistics.mean(r['snapshots'][-1]['identical_incidence_pairs']['cosine']['mean'] for r in runs),
                'minimum_update_pair_cosine': min(r['snapshots'][-1]['actual_update_pairs']['cosine']['minimum'] for r in runs),
                'maximum_tied_global_aggregate_relative_l2': max(r['matched_final_aggregate_fidelity']['global_relative_l2'] for r in runs),
                'maximum_absolute_test_loss_change': max(abs(r['heldout']['test']['delta']['loss_relative_change']) for r in runs),
                'maximum_absolute_test_discrimination_change': max(abs(r['heldout']['test']['delta']['discrimination_change']) for r in runs),
                'test_token_coverage': sorted({r['heldout']['test']['token_coverage'] for r in runs})})
    capacity = []
    for order in (4, 8, 12):
        seed_rows = []
        for seed in (3, 7, 11):
            chosen = {row['family']: row for row in codecs['validation_selected']
                      if row['order'] == order and row['seed'] == seed and row['loss_budget'] == .001}
            if any(chosen[family]['selected'] is None for family in ('direct', 'group', 'hybrid')):
                raise ValueError('primary quality budget has an infeasible family')
            selections = {family: row['selected'] for family, row in chosen.items()}
            variants = {(r['name']): r for r in codecs['rows'] if r['order'] == order and r['seed'] == seed}
            timed = {family(r['name']): r
                     for r in query['runs'] if r['order'] == order and r['seed'] == seed}
            trained = next(r for r in training['runs'] if r['convention'] == 'legacy' and
                           r['order'] == order and r['seed'] == seed)
            for kind, r in timed.items():
                selected = selections[kind]
                if (r['name'] != selected['name'] or r['model_bytes'] != selected['model_bytes'] or
                        r['checkpoint_sha256'] != trained['checkpoint_sha256']):
                    raise ValueError('query differs from selected trained checkpoint/codec')
            seed_rows.append({'seed': seed, 'selected': selections,
                'extra_complete_saving': 1 - selections['group']['model_bytes'] / selections['direct']['model_bytes'],
                'relative_l2': {family: variants[item['name']]['known_aggregate_relative_l2'] for family, item in selections.items()},
                'new_known_dynamic_p50_ms': {family: r['query']['known']['timing']['dynamic_new']['p50_ms'] for family, r in timed.items()},
                'new_oov_dynamic_p50_ms': {family: r['query']['oov']['timing']['dynamic_new']['p50_ms'] for family, r in timed.items()},
                'oov_coverage': {family: {k: r['query']['oov']['fidelity'][k] for k in ('accepted', 'total')}
                                 for family, r in timed.items()},
                'primary_test_point_gates_pass': all(item['test_quality_budget_passed'] for item in selections.values()),
                'test_block_intervals': {family: chosen[family].get('test_paired_block_interval') for family in chosen}})
        capacity.append({'order': order, 'seeds': seed_rows,
            'median_extra_complete_saving': median(r['extra_complete_saving'] for r in seed_rows),
            'median_model_bytes': {family: median(r['selected'][family]['model_bytes'] for r in seed_rows)
                                   for family in ('direct', 'group', 'hybrid')},
            'median_known_dynamic_p50_ms': {family: median(r['new_known_dynamic_p50_ms'][family] for r in seed_rows)
                                            for family in ('direct', 'group', 'hybrid')},
            'all_primary_test_point_gates_pass': all(r['primary_test_point_gates_pass'] for r in seed_rows)})
    native_times = {k: value['p50_ms'] for k, value in native['timing'].items()}
    return {'training_pairs': len(training['runs']), 'trained_codec_checkpoints': codecs['runs'],
        'scored_codec_variants': codecs['variants'], 'dynamics': dynamics, 'primary_capacity': capacity,
        'cross_corpus_query_retiming': query_summary(query),
        'native': {'original_rows': native['partition']['original_rows'], 'bags': native['plan']['bags'],
            'p50_ms': native_times, 'compile_speedup': native_times['compile_old'] / native_times['compile_new'],
            'total_query_speedup': native_times['dynamic_old'] / native_times['dynamic_new'],
            'plan_arrays_byte_identical': True, 'outputs_byte_identical': True},
        'scope': 'Fixed finite SGNS corpus prefix and registered known-word domain; native retiming reuses the older large enwik8 table.',
        'limitations': ['Shared warm CPU; per-checkpoint query timings are descriptive across fresh workers.',
            'Group projection/quantization changes original rows; original-row byte-preserving storage is separate.',
            'OOV hybrid costs and narrower precomputed serving controls remain in the source analysis.',
            'No large grouped-model retraining, convergence, semantic, LLM quality or GPU claim.']}


def figures(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    orders = [r['order'] for r in summary['primary_capacity']]
    for family, offset, color in (('direct', -.18, '#356bb3'), ('group', .18, '#db7e1a')):
        axes[0].bar([i + offset for i in range(3)],
                    [r['median_model_bytes'][family] / 2**20 for r in summary['primary_capacity']],
                    width=.36, color=color, label=family)
        axes[1].bar([i + offset for i in range(3)],
                    [r['median_known_dynamic_p50_ms'][family] for r in summary['primary_capacity']],
                    width=.36, color=color, label=family)
    for ax in axes:
        ax.set_xticks(range(3), orders)
        ax.set_xlabel('Maximum n-gram order')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Complete model (MiB)')
    axes[1].set_ylabel('512 known bags: compilation + query (ms)')
    axes[0].legend()
    fig.suptitle('enwik8 prefix: fixed 0.1% SGNS loss / 5% aggregate-error budget')
    fig.text(.5, .015, 'Three seeds; validation-selected codecs. Certified known-word domain, shared warm CPU. Original row bytes change.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save_figure(fig, output / 'capacity-and-query')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    native = summary['native']['p50_ms']
    for label, offset, color, keys in (('Previous', -.18, '#356bb3', ('compile_old', 'dynamic_old')),
                                     ('Fused', .18, '#db7e1a', ('compile_new', 'dynamic_new'))):
        ax.bar([i + offset for i in range(2)], [native[k] for k in keys], width=.36, color=color, label=label)
    ax.set_xticks(range(2), ['Plan compilation', 'Compilation + native query'])
    ax.set_ylabel('p50 (ms), 512 registered bags')
    ax.set_title('1.86M trained rows: identical plans and native output bytes')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.text(.5, .02, 'Paired shared-CPU run; 4 MiB decoded cache; no new quality or raw-mmap comparison.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_figure(fig, output / 'native-total-query')
    plt.close(fig)


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=160)
    svg = path.with_suffix('.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'codecs', 'query', 'native'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frozen-query', type=Path, help='Separate previous-corpus frozen query comparison')
    p.add_argument('--plots', action='store_true')
    cfg = p.parse_args()
    inputs = {name: getattr(cfg, name).read_bytes() for name in ('training', 'codecs', 'query', 'native')}
    reports = {name: json.loads(value) for name, value in inputs.items()}
    if reports['codecs']['report_sha256'] != reports['query']['frozen_report_sha256']:
        raise ValueError('codec analysis and retiming must use the same frozen report')
    summary = summarize(**reports)
    if cfg.frozen_query:
        inputs['frozen_query'] = cfg.frozen_query.read_bytes()
        summary['previous_corpus_frozen_query'] = query_summary(json.loads(inputs['frozen_query']))
    summary['inputs_sha256'] = {name: hashlib.sha256(value).hexdigest() for name, value in inputs.items()}
    summary['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if cfg.plots:
        figures(summary, cfg.output_dir)


if __name__ == '__main__':
    main()
