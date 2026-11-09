"""Stdlib audit of archived provenance, counts and summary arithmetic.

This does not reconstruct unpublished trained rows, rerun cosine measurements,
or independently verify the logged native reconstruction assertions.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
COUNT_FIELDS = ('sampled_pairs', 'valid_pairs', 'undefined_zero_norm_pairs',
                'ge_0_99_pairs', 'joint_0_99_pairs')
DIAGNOSTICS = (
    'ancestor_frequency_holding_lengths_and_child_bin',
    'child_frequency_holding_lengths_and_ancestor_bin',
    'child_length_holding_ancestor_length_and_frequency_bins',
    'coverage_fraction_holding_lengths',
    'identical_incidence_frequency_holding_lengths')


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def counts(parts, total):
    for field in COUNT_FIELDS:
        assert sum(p[field] for p in parts) == total[field], field


def audit_tree(value):
    if isinstance(value, list):
        for v in value:
            audit_tree(v)
    elif isinstance(value, dict):
        if set(COUNT_FIELDS) <= value.keys():
            n, valid, zero, high, joint = [value[k] for k in COUNT_FIELDS]
            assert all(isinstance(v, int) and v >= 0 for v in (n, valid, zero, high, joint))
            assert valid + zero == n and joint <= high <= valid
            for name in ('cosine', 'norm_ratio', 'replacement_relative_l2'):
                assert value[name]['n'] == valid
        if 'weighted_spearman' in value:
            groups = value['groups']
            assert len({g['stratum'] for g in groups}) == len(groups)
            measured = [g for g in groups if g['spearman'] is not None]
            for g in measured:
                assert g['valid_pairs'] >= 50 and -1.000000000001 <= g['spearman'] <= 1.000000000001
            n = sum(g['valid_pairs'] for g in measured)
            assert value['measured_pairs'] == n
            close(value['weighted_spearman'],
                  sum(g['valid_pairs'] * g['spearman'] for g in measured) / n if n else None)
            if 'stratum_values' in value:
                assert {g['stratum'] for g in groups} == set(range(len(value['stratum_values'])))
        for v in value.values():
            audit_tree(v)


def relation(r, rows, primary):
    audit_tree(r)
    assert r['sampled_pairs'] == r['source']['sampled_pairs'] <= r['population_pairs']
    assert r['sample_is_complete'] == (r['sampled_pairs'] == r['population_pairs'])
    assert r['matched_source']['sampled_pairs'] == r['matched_control']['sampled_pairs']
    assert r['unmatched_pairs'] + r['matched_source']['sampled_pairs'] == r['sampled_pairs']
    counts([c['structural'] for c in r['cells']], r['source'])
    counts([c['matched_control'] for c in r['cells']], r['matched_control'])
    if primary:
        counts([r['identical_incidence'], r['different_incidence']], r['source'])
    for name in ('cosine_0_99', 'cosine_and_norm_0_99'):
        c = r[name + '_observed_coverage']
        assert 0 <= c['unique_rows_in_sample'] <= c['total_source_rows'] == rows
        assert 0 <= c['occurrence_mass'] <= c['total_source_occurrence_mass']
        assert c['full_population_coverage'] == r['sample_is_complete']
    if primary:
        assert r['relation'] in ('parent', 'ancestor') and r['sample_is_complete']
        a = r['ancestry_diagnostics']
        counts([c['source'] for c in a['coverage_cells']], r['source'])
        assert a['child_anchored_source']['sampled_pairs'] == a['child_anchored_control']['sampled_pairs']
        assert a['child_anchored_unmatched_pairs'] + a['child_anchored_source']['sampled_pairs'] == r['sampled_pairs']
        b = a['best_per_child_matched_candidate_count']
        n = b['candidate_rows']
        assert n <= rows and b['source_best_cosine']['n'] == b['control_best_cosine']['n'] == n
        for side in ('source', 'control'):
            assert 0 <= b[side + '_joint_0_99_rows'] <= b[side + '_ge_0_99_rows'] <= n
    else:
        assert r['relation'] in ('prefix', 'suffix') and r['sampled_pairs'] <= 50000


def storage(d):
    n, dim = d['rows'], d['dimension']
    s = d['storage']; identity = s['identity']
    assert identity['format_version'] == 3 and identity['row_map_bytes'] == 0
    assert identity['unique_rows'] == identity['n_rows'] == n
    assert identity['input_bytes'] == n * dim * 4
    assert {a['minimum_norm_ratio'] for a in s['approximate']} == {.99, .95}
    for a in s['approximate']:
        p, f = a['packed'], a['fidelity']
        assert a['retained_rows'] + a['merged_rows'] == n
        assert p['unique_rows'] == p['marked_rows'] == a['retained_rows']
        assert p['n_rows'] == f['rows'] == n and p['row_map_bytes'] == n * 8
        assert p['format_version'] == 2 and p['input_bytes'] == n * dim * 4
        assert 0 <= a['changed_occurrence_mass'] <= a['total_occurrence_mass']
        assert a['native_reads_approximation_byte_exact'] is True
        assert len(a['approximation_sha256']) == 64
        assert a['minimum_cosine'] == .99 and f['minimum_cosine'] >= .99 - 1e-7
        assert f['max_relative_norm_change'] <= 1 / a['minimum_norm_ratio'] - 1 + 1e-7


def summary(root=ROOT):
    tiny_path = root.parent / 'native_ngram_evidence_20261018/trained-reference.json'
    en_path = root.parent / 'cross_corpus_evidence_20261102/training-report.json'
    original = {'tinyshakespeare': load(tiny_path), 'enwik8': load(en_path)}
    original_paths = {'tinyshakespeare': tiny_path, 'enwik8': en_path}
    runs = []
    auxiliary_pairs = auxiliary_high = 0
    for dataset, number in (('tinyshakespeare', 9), ('enwik8', 18), ('qwen-auxiliary', 6)):
        folder = root / dataset; report = load(folder / 'report.json')
        assert report['status'] == 'complete' and len(report['runs']) == number
        assert {p.name for p in folder.iterdir() if p.is_dir()} == {r['name'] for r in report['runs']}
        env = report['environment']; snapshot = root / 'runtime' / env['commit']
        assert env['gpu'] == 'unused' and env['threads'] == env['interop_threads'] == 1
        assert env['source_sha256'] == digest(snapshot / 'tools/analyze_conditional_similarity.py')
        assert env['analysis_source_sha256'] == digest(snapshot / 'tools/conditional_similarity.py')
        assert report['protocol_sha256'] == digest(snapshot / 'docs/conditional_similarity_protocol_20261109.md')
        for item in report['runs']:
            path = folder / item['name'] / 'analysis.json'
            assert digest(path) == item['analysis_sha256']
            d = load(path)
            if dataset == 'qwen-auxiliary':
                previous_path = root.parent / 'qwen_native_identity_evidence_20261109/results' / item['name'] / 'manifest.json'
                previous = load(previous_path)
                assert d['source_fixture_manifest_sha256'] == digest(previous_path)
                assert d['source_row_sha256'] == previous['source_row_sha256']
                assert d['vocabulary_sha256'] == previous['source_vocabulary_sha256']
                assert len(d['heads']) == 6
                for i, h in enumerate(d['heads']):
                    assert h['head'] == i and h['order'] == 2 + i // 2
                    assert {r['relation'] for r in h['relations']} == {'prefix', 'suffix'}
                    for r in h['relations']:
                        relation(r, h['rows'], False)
                        auxiliary_pairs += r['sampled_pairs']
                        auxiliary_high += r['source']['ge_0_99_pairs']
                continue
            assert d['dataset'] == dataset and d['order'] in (4, 8, 12) and d['seed'] in (3, 7, 11)
            assert d['source_training_report_sha256'] == digest(original_paths[dataset])
            src = original[dataset]
            if dataset == 'tinyshakespeare':
                old = next(r for r in src['runs'] if r['max_order'] == d['order'] and r['seed'] == d['seed'])
                assert d['corpus_sha256'] == src['corpus_sha256']
                assert d['source_row_sha256'] == old['learned_row_sha256']
                assert d['rows'] == old['ngram_rows'] and d['convention'] == 'legacy'
            else:
                old = next(r for r in src['runs'] if all(r[k] == d[k] for k in ('convention', 'order', 'seed')))
                assert d['corpus_sha256'] == src['tokenized_corpus_sha256']
                assert d['source_checkpoint_sha256'] == old['checkpoint_sha256']
                assert d['sampled_center_events_sha256'] == old['sampled_center_events_sha256']
            assert [r['relation'] for r in d['relations']] == ['parent', 'ancestor']
            for r in d['relations']:
                relation(r, d['rows'], True)
            storage(d)
            parent, ancestor = d['relations']; diagnostics = ancestor['ancestry_diagnostics']
            record = {'dataset': dataset, 'convention': d['convention'], 'order': d['order'], 'seed': d['seed'],
                      'rows': d['rows'], 'ancestor_pairs': ancestor['sampled_pairs'],
                      'parent_pair99_fraction': parent['source']['ge_0_99_pairs'] / parent['sampled_pairs'],
                      'ancestor_pair99_fraction': ancestor['source']['ge_0_99_pairs'] / ancestor['sampled_pairs'],
                      'child_anchored_source99_fraction': diagnostics['child_anchored_source']['ge_0_99_pairs'] / diagnostics['child_anchored_source']['sampled_pairs'],
                      'child_anchored_control99_fraction': diagnostics['child_anchored_control']['ge_0_99_pairs'] / diagnostics['child_anchored_control']['sampled_pairs'],
                      'conditional_rho': {k: diagnostics[k]['weighted_spearman'] for k in DIAGNOSTICS},
                      'best_per_child': diagnostics['best_per_child_matched_candidate_count'], 'capacity': []}
            for a in d['storage']['approximate']:
                record['capacity'].append({'norm_ratio': a['minimum_norm_ratio'],
                    'native_byte_saving_fraction': 1 - a['packed']['artifact_bytes'] / d['storage']['identity']['artifact_bytes'],
                    'merged_row_fraction': a['merged_rows'] / d['rows'],
                    'changed_occurrence_fraction': a['changed_occurrence_mass'] / a['total_occurrence_mass'],
                    'weighted_relative_frobenius': a['fidelity']['weighted_relative_frobenius']})
            runs.append(record)
    assert len({(r['dataset'], r['convention'], r['order'], r['seed']) for r in runs}) == 27
    medians = []
    for dataset, convention in (('tinyshakespeare', 'legacy'), ('enwik8', 'legacy'), ('enwik8', 'average')):
        for order in (4, 8, 12):
            selected = [r for r in runs if (r['dataset'], r['convention'], r['order']) == (dataset, convention, order)]
            assert {r['seed'] for r in selected} == {3, 7, 11}
            m = {'dataset': dataset, 'convention': convention, 'order': order}
            for k in ('parent_pair99_fraction', 'ancestor_pair99_fraction', 'child_anchored_source99_fraction', 'child_anchored_control99_fraction'):
                m[k] = statistics.median(r[k] for r in selected)
            m['conditional_rho'] = {k: statistics.median(r['conditional_rho'][k] for r in selected) for k in DIAGNOSTICS}
            m['capacity'] = [{'norm_ratio': q, 'native_byte_saving_fraction': statistics.median(next(a['native_byte_saving_fraction'] for a in r['capacity'] if a['norm_ratio'] == q) for r in selected)} for q in (.99, .95)]
            medians.append(m)
    return {'status': 'verified', 'primary_character_models': 27, 'primary_relations': 54,
            'auxiliary_qwen_tables': 6, 'auxiliary_sampled_sibling_pairs': auxiliary_pairs,
            'auxiliary_sibling_cosine_ge_0_99_pairs': auxiliary_high,
            'capacity_scope': 'Approximate n-gram-only native table versus v3 identity; excludes key registry, word/output weights and task quality',
            'seed_medians': medians, 'runs': runs}


def verify(root=ROOT):
    files = load(root / 'manifest.json')['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(files), actual ^ set(files)
    for name, info in files.items():
        p = root / name
        assert p.stat().st_size == info['bytes'] and digest(p) == info['sha256'], name
    result = summary(root)
    assert result == load(root / 'summary.json')
    assert result['auxiliary_sampled_sibling_pairs'] == 3600000
    assert result['auxiliary_sibling_cosine_ge_0_99_pairs'] == 7
    print('Verified 27 character models, 54 complete ancestor/parent analyses and 6 auxiliary Qwen tables; archive/provenance/arithmetic only.')


if __name__ == '__main__':
    verify()
