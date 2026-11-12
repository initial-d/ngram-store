"""Verify the complete independent-key experiment without GPUs or model weights.

The default verifies provenance, paired budgets, preflights and point estimates
from report scalars. --bootstrap additionally checks per-case arrays and exactly
reproduces all eight intervals with NumPy; no Torch dependency is required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = ('rewrite', 'paraphrase', 'neighborhood')
SEEDS = (3, 7, 11)
SOURCE = '425c1876614b9533ecf3b5e092e1dbed0bfd7651'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=0., abs_tol=1e-12), (a, b)


def harmonic(values):
    return 3 / math.fsum(1 / v for v in values) if all(values) else 0.


def verify(root=ROOT, bootstrap=False):
    manifest = load(root / 'manifest.json')['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(manifest), 'Archive file set changed'
    for name, info in manifest.items():
        path = root / name
        assert path.stat().st_size == info['bytes'] and digest(path) == info['sha256'], name
    analysis = load(root / 'final-analysis.json')
    assert analysis['status'] == 'complete-audited' and analysis['completed_trained_runs'] == 24
    assert analysis['source_sha'] == SOURCE
    assert analysis['analyzer_sha256'] == digest(root / 'analyze.py')
    assert analysis['paired_test_arrays_sha256'] == digest(root / 'paired-test-arrays.npz')
    assert analysis['bootstrap_replicates'] == 2000 and analysis['bootstrap_rng_seed'] == 20261106
    protocol = load(root / 'ngram_identity_v1.json')
    assert protocol['identity_control'] and not protocol['train_decoder_suffix']
    assert protocol['seeds'] == list(SEEDS) and protocol['epochs'] == 20
    queue = load(root / 'queue.json')
    assert queue['status'] == 'complete' and len(queue['jobs']) == 26
    assert all(j['exit_code'] == 0 for j in queue['jobs'])
    expected = {f'{m}-{n}-{v}-s{s}' for m in ('qwen06', 'qwen35')
                for n in (1024, 20877) for v in ('exact', 'hash') for s in SEEDS}
    reports = {}
    for file in sorted((root / 'reports/runs').glob('*/report.json')):
        r = load(file)
        name = file.parent.name
        assert digest(file) == analysis['input_sha256'][f'runs/{name}/report.json']
        assert r['status'] == 'complete' and r['environment']['code_sha'] == SOURCE
        assert r['protocol'] == protocol and r['protocol_sha256'] == digest(root / 'ngram_identity_v1.json')
        assert not any(r['loading'].values()) and r['environment']['tf32'] is False
        for source_name, source_sha in r['environment']['source_hashes'].items():
            assert digest(root / 'runtime' / SOURCE / 'tools' / source_name) == source_sha
        model, count, variant, seed = name.split('-')
        assert r['variant'] == variant and r['seed'] == int(seed[1:])
        count = int(count)
        vocab = load(root / f'reports/vocab-{model}-{count}/report.json')
        assert vocab['status'] == 'built' and vocab['metadata']['training_sequences'] == count
        assert vocab['vocabulary_sha256'] == r['identity']['vocabulary_sha256']
        assert vocab['prepared_sha256'] == r['prepared_sha256']
        assert len(r['epochs']) == 20 and [e['epoch'] for e in r['epochs']] == list(range(20))
        assert all(e['actual_sequences'] == count for e in r['epochs'])
        assert r['optimizer_steps'] == r['epochs'][-1]['steps']
        assert r['trainable_parameters'] == r['memory_parameters']
        assert all('memory' in p for p in r['trained_parameter_names'])
        assert r['identity']['row_bytes'] == r['physical_rows'] * 32 * 4
        assert r['replay_check']['logits_byte_exact'] and r['replay_check']['all_masked_gradients_exact']
        for order, collision in r['identity']['collision_audit'].items():
            unique = r['identity']['unique_keys_per_order'][order]
            assert collision['unique_training_keys'] == unique
            assert 0 < collision['distinct_joint_head_addresses'] <= unique
            assert len(collision['heads']) == 2
            assert all(0 <= h['keys_in_shared_buckets'] <= unique for h in collision['heads'])
        for split, metrics in r['metrics'].items():
            cases = count if split == 'all' else ((count + 9) // 10 if split == 'calibration' else count - (count + 9) // 10)
            for group in GROUPS:
                assert metrics[group]['cases'] == cases
                assert 0 <= metrics[group]['success'] <= 1
            close(harmonic([metrics[g]['success'] for g in GROUPS]), metrics['harmonic_score'])
        if variant == 'hash':
            assert r['native_oov_control']['same_trained_delta']
            assert r['native_oov_control']['case_scores_sha256'] == analysis['input_sha256'][f'runs/{name}/case-scores-native-oov.json']
        reports[name] = r
    assert set(reports) == expected
    assert {r['run'] for r in analysis['runs']} == expected
    for item in analysis['runs']:
        report = reports[item['run']]
        assert item['metrics'] == report['metrics']
        assert item['identity_bundle_bytes'] == report['identity_bundle_bytes']
        if report['variant'] == 'hash':
            assert item['native_oov_metrics'] == report['native_oov_control']['metrics']
    for model in ('qwen06', 'qwen35'):
        p = load(root / f'reports/{model}-preflight/report.json')
        assert p['status'] == 'passed' and p['backbone_unchanged_after_optimizer']
        for k in ('paired_training_logits_byte_exact', 'initial_nonrow_gradients_byte_exact',
                  'all_training_key_vectors_initially_byte_exact', 'trainable_parameter_budget_matched'):
            assert p[k]
        assert p['gradient_pooling_max_absolute_error'] == 0
        assert len(p['question_requests']) == 16 and all(q['byte_exact'] for q in p['question_requests'])
        assert all(c['logits_byte_exact'] and c['all_masked_gradients_exact'] for c in p['masked_replay'])
    if bootstrap:
        import numpy as np
        data = np.load(root / 'paired-test-arrays.npz', allow_pickle=False)
        source = (root / 'runtime' / SOURCE / 'tools/counterfact_analyze.py').read_text()
        namespace = {'np': np, 'GROUPS': GROUPS}
        exec(source[source.index('def harmonic('):source.index('def main(')], namespace)
    comparisons = analysis['exact_vs_control']
    assert len(comparisons) == 8
    assert {(c['model'], c['edits'], c['control']) for c in comparisons} == {
        (m, n, v) for m in ('qwen06', 'qwen35') for n in (1024, 20877)
        for v in ('hash', 'hash-native-oov')}
    for c in comparisons:
        assert c['seeds'] == list(SEEDS) and c['paired_seeds'] == 3
        aa, bb, am, bm = [], [], [], []
        for seed in SEEDS:
            an = f"{c['model']}-{c['edits']}-exact-s{seed}"
            bn = f"{c['model']}-{c['edits']}-hash-s{seed}"
            a, b = reports[an], reports[bn]
            for field in ('protocol_sha256', 'prepared_sha256', 'trainable_parameters',
                          'memory_parameters', 'physical_rows', 'optimizer_steps',
                          'frozen_weight_sha256', 'frozen_decoder_suffix_sha256'):
                assert a[field] == b[field], field
            assert a['identity']['row_bytes'] == b['identity']['row_bytes']
            br = b['metrics'] if c['control'] == 'hash' else b['native_oov_control']['metrics']
            av = [a['metrics']['test'][g]['success'] for g in GROUPS]
            bv = [br['test'][g]['success'] for g in GROUPS]
            am.append(av); bm.append(bv)
            if bootstrap:
                assert np.array_equal(data[an + '_ids'], data[bn + '_ids'])
                assert len(set(data[an + '_ids'].tolist())) == c['paired_cases']
                va = data[an + '_success']
                vb = data[bn + ('_success' if c['control'] == 'hash' else '_native_success')]
                assert va.shape == vb.shape == (c['paired_cases'], 3)
                assert np.isfinite(va).all() and np.isfinite(vb).all()
                assert ((va >= 0) & (va <= 1)).all() and ((vb >= 0) & (vb <= 1)).all()
                assert np.allclose(va.mean(0), av, atol=1e-12, rtol=0)
                assert np.allclose(vb.mean(0), bv, atol=1e-12, rtol=0)
                aa.append(va); bb.append(vb)
        av = [math.fsum(row[i] for row in am) / 3 for i in range(3)]
        bv = [math.fsum(row[i] for row in bm) / 3 for i in range(3)]
        close(harmonic(av) - harmonic(bv), c['harmonic_delta'])
        for i, group in enumerate(GROUPS):
            close(av[i] - bv[i], c['group_deltas'][group]['delta'])
        ns = c['group_deltas']['neighborhood']
        positive = c['harmonic_delta_interval95'][0] > 0
        assert c['gate_positive_H_with_NS_point_within_2pp'] == (positive and ns['delta'] >= -.02)
        assert c['gate_positive_H_with_NS_interval_within_2pp'] == (positive and ns['interval95'][0] >= -.02)
        if bootstrap:
            computed = namespace['paired_interval'](np.stack(aa), np.stack(bb), 2000)
            assert computed['harmonic_delta_interval95'] == c['harmonic_delta_interval95']
            assert computed['group_deltas'] == c['group_deltas']
    if bootstrap:
        data.close()
    return len(reports), len(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()
    runs, pairs = verify(bootstrap=args.bootstrap)
    print(f'Verified {runs} completed runs, 2 preflights, source hashes, matched budgets and {pairs} comparisons.')
    print('All eight intervals reproduced from paired test arrays.' if args.bootstrap else
          'Intervals are hash-archived; use --bootstrap with NumPy to reproduce them. Loss-level scalars and deltas remain remote.')


if __name__ == '__main__':
    main()
