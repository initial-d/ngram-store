"""Recompute archived MultiCounterFact scalars without Torch or model weights.

Default checks are dependency-free. --bootstrap also reproduces every archived
paired interval using NumPy and the source-pinned analysis implementation.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = ('rewrite', 'paraphrase', 'neighborhood')
SEEDS = (3, 7, 11)
VARIANTS = ('small', 'mid', 'large', 'head-constant', 'off', 'adapter')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data), data


def harmonic(values):
    return 3 / math.fsum(1 / v for v in values) if all(values) else 0.


def summarize(cases):
    result = {}
    for group in GROUPS:
        rows = [c['groups'][group] for c in cases]
        result[group] = dict(
            cases=len(cases), prompts=sum(map(len, rows)),
            success=math.fsum(sum(p['success'] for p in row) / len(row) for row in rows) / len(rows),
            teacher_forced_accuracy=math.fsum(sum(p['correct'] for p in row) / len(row) for row in rows) / len(rows))
    result['harmonic_score'] = harmonic([result[g]['success'] for g in GROUPS])
    return result


def check_cases(cases, metrics):
    assert len(cases) == 1024 and len({c['case_id'] for c in cases}) == 1024
    assert [c['case_index'] for c in cases] == list(range(1024))
    for c in cases:
        assert c['split'] == ('calibration' if c['case_index'] % 10 == 0 else 'test')
        for group in GROUPS:
            assert c['groups'][group]
            for p in c['groups'][group]:
                assert all(math.isfinite(p[k]) and p[k] >= 0 for k in ('target_new', 'target_true'))
                assert p['success'] == (p['target_true'] < p['target_new'] if group == 'neighborhood' else p['target_new'] < p['target_true'])
    for split in ('all', 'calibration', 'test'):
        selected = [c for c in cases if split == 'all' or c['split'] == split]
        computed = summarize(selected)
        for group in GROUPS:
            assert computed[group] == metrics[split][group]
        assert math.isclose(computed['harmonic_score'], metrics[split]['harmonic_score'], abs_tol=1e-14)
    assert metrics['test']['rewrite']['cases'] == 921


def check_run(folder):
    r, raw = load(folder / 'report.json')
    assert r['status'] == 'complete'
    assert r['environment']['transformers'] == '5.3.0'
    assert 'A100' in r['environment']['gpu'] and not any(r['loading'].values())
    assert r['environment']['tf32'] is False  # Third-party FLA TF32 is disclosed separately.
    source = ROOT / 'runtime' / r['environment']['code_sha']
    for name, expected in r['environment']['source_hashes'].items():
        assert digest((source / 'tools' / name).read_bytes()) == expected
    protocol_name = 'counterfact_memory_only_probe_v1.json' if folder.parent.name == 'probe-runs' else 'counterfact_capacity_v1.json'
    assert digest((source / 'benchmarks' / protocol_name).read_bytes()) == r['protocol_sha256']
    if 'selected_top_k' in r:
        assert r['selection_fold'] == 'calibration'
        for k, candidate in r['candidates'].items():
            cases, data = load(folder / f'case-scores-k{k}.json.gz')
            assert digest(data) == candidate['case_scores_sha256']
            check_cases(cases, candidate['metrics'])
        selected = max((1, 3), key=lambda k: (r['candidates'][str(k)]['metrics']['calibration']['harmonic_score'], -k))
        assert selected == r['selected_top_k']
        assert r['metrics'] == r['candidates'][str(selected)]['metrics']
        cases, data = load(folder / f'case-scores-k{selected}.json.gz')
    else:
        cases, data = load(folder / 'case-scores.json.gz')
        expected = r.get('case_scores_sha256') or r['score_artifacts']['case-scores.json']['sha256']
        assert digest(data) == expected
        check_cases(cases, r['metrics'])
    if 'seed' in r:
        assert r['replay_check']['logits_byte_exact'] and r['replay_check']['all_masked_gradients_exact']
        assert len(r['epochs']) == r['protocol']['epochs']
        assert all(epoch['actual_sequences'] == 1024 for epoch in r['epochs'])
        assert r['optimizer_steps'] == r['epochs'][-1]['steps']
    return r, raw, cases, data


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-13), (a, b)


def check_analysis(folder, name, bootstrap):
    analysis, _ = load(ROOT / name)
    runs = {}
    for path in sorted((ROOT / folder).glob('*/report.json')):
        runs[path.parent.name] = check_run(path.parent)
    assert analysis['status'] == 'partial' and analysis['expected_trained_runs'] == 72
    assert analysis['completed_trained_runs'] == (36 if folder == 'runs' else 4)
    assert len(runs) == (40 if folder == 'runs' else 4)
    if folder == 'runs':
        expected = {f'{m}-1024-{v}-s{s}' for m in ('qwen06', 'qwen35') for v in VARIANTS for s in SEEDS}
        expected |= {f'{m}-1024-{v}' for m in ('qwen06', 'qwen35') for v in ('unedited-base', 'bm25')}
    else:
        expected = {f'{m}-1024-{v}-s3' for m in ('qwen06', 'qwen35') for v in ('large', 'adapter')}
    assert set(runs) == expected and {x['run'] for x in analysis['runs']} == expected
    for item in analysis['runs']:
        report, raw, cases, data = runs[item['run']]
        assert digest(raw) == item['report_sha256'] and digest(data) == item['case_scores_sha256']
        assert report['metrics'] == item['metrics']
        near = sum(abs(p['target_true'] - p['target_new']) <= .0014 for c in cases for rows in c['groups'].values() for p in rows)
        assert near == item['near_ties_at_0_0014_nll']
        safe = [p for c in cases if c['split'] == 'test' for p in c['groups']['neighborhood'] if not p['prompt_matches_an_edited_rewrite']]
        assert len(safe) == item['unedited_neighbor_diagnostic']['prompts']
        close(sum(p['success'] for p in safe) / len(safe), item['unedited_neighbor_diagnostic']['success'])
    assert len(analysis['large_vs_control']) == (10 if folder == 'runs' else 2)
    for c in analysis['large_vs_control']:
        assert c['seeds'] == ([3, 7, 11] if folder == 'runs' else [3])
        av, bv, aa, bb = [], [], [], []
        for seed in c['seeds']:
            ar = runs[f"{c['model']}-1024-large-s{seed}"]
            br = runs[f"{c['model']}-1024-{c['control']}-s{seed}"]
            av.append([ar[0]['metrics']['test'][g]['success'] for g in GROUPS])
            bv.append([br[0]['metrics']['test'][g]['success'] for g in GROUPS])
            ac = sorted((x for x in ar[2] if x['split'] == 'test'), key=lambda x: x['case_index'])
            bc = sorted((x for x in br[2] if x['split'] == 'test'), key=lambda x: x['case_index'])
            assert [x['case_id'] for x in ac] == [x['case_id'] for x in bc]
            aa.append([[sum(p['success'] for p in x['groups'][g]) / len(x['groups'][g]) for g in GROUPS] for x in ac])
            bb.append([[sum(p['success'] for p in x['groups'][g]) / len(x['groups'][g]) for g in GROUPS] for x in bc])
        am = [math.fsum(row[i] for row in av) / len(av) for i in range(3)]
        bm = [math.fsum(row[i] for row in bv) / len(bv) for i in range(3)]
        close(harmonic(am) - harmonic(bm), c['harmonic_delta'])
        for i, group in enumerate(GROUPS):
            close(am[i] - bm[i], c['group_deltas'][group]['delta'])
        if bootstrap:
            import numpy as np
            # Execute the archived pure analysis functions, without importing Torch.
            sha = 'b81bfaae14af2dc12acb44e6df1d377f35e6a2f5'
            source = (ROOT / 'runtime' / sha / 'tools/counterfact_analyze.py').read_text()
            namespace = {'np': np, 'GROUPS': GROUPS}
            exec(source[source.index('def harmonic('):source.index('def main(')], namespace)
            reproduced = namespace['paired_interval'](np.array(aa), np.array(bb))
            assert reproduced['harmonic_delta_interval95'] == c['harmonic_delta_interval95']
            assert reproduced['group_deltas'] == c['group_deltas']
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()
    manifest, _ = load(ROOT / 'manifest.json')
    for name, info in manifest['files'].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == info['bytes'] and digest(raw) == info['sha256']
    check_analysis('runs', 'small-analysis.json', args.bootstrap)
    probe_runs = check_analysis('probe-runs', 'probe-analysis.json', args.bootstrap)
    for model in ('qwen06', 'qwen35'):
        r, _ = load(ROOT / f'native-{model}-probe/report.json')
        source, raw, _, _ = probe_runs[f'{model}-1024-large-s3']
        assert r['status'] == 'passed' and r['all_original_rows_byte_exact']
        assert r['independent_native_bundle_reopen_byte_exact']
        assert digest(raw) == r['source_run_report_sha256']
        assert source['delta']['sha256'] == r['source_delta_sha256']
        assert source['protocol_sha256'] == r['protocol_sha256']
        assert r['table']['shape'] == [917516, 32] and sum(r['table']['head_sizes']) == 917516
        assert r['table']['input_bytes'] == 917516 * 32 * 4
        assert r['table']['artifact_bytes'] > r['table']['input_bytes']
        assert r['bundle_bytes'] > r['source_delta_bytes']
        assert len(r['requests']) == 16 and all(x['byte_exact'] for x in r['requests'])
        assert r['staging_stats']['staging_bytes'] <= r['staging_stats']['staging_byte_limit'] == 1048576
    print('Verified 40 small-grid runs, 4 exploratory probes, RAG calibration selection, all scalar scores, paired point estimates and 2 native reopen reports.')
    print('Full 20,877-edit grid is not included; native fidelity passed but both artifacts expanded.')
    if args.bootstrap:
        print('Also reproduced all 12 paired bootstrap intervals exactly.')
    else:
        print('Bootstrap intervals are hash-archived; use --bootstrap with NumPy to recompute them.')


if __name__ == '__main__':
    main()
