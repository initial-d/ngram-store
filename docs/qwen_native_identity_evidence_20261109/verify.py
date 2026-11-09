"""Audit published cost scalars and paired logit hashes without Torch or GPUs.

Original vector bytes, token keys and model weights remain in the source runs;
rerun the benchmark tools with those weights to recheck the row/logit assertions.
This verifier checks archive integrity, source provenance, arithmetic and pairs.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'ngram_identity_evidence_20261107/reports/runs'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(samples, percent):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def summary(root=ROOT):
    expected = {f'{m}-20877-exact-s{s}' for m in ('qwen06', 'qwen35') for s in (3, 7, 11)}
    assert {d.name for d in (root / 'results').iterdir()} == expected
    runs = []
    pairs = 0
    for name in sorted(expected):
        folder = root / 'results' / name
        source = load(PREVIOUS / name / 'report.json')
        manifest = load(folder / 'manifest.json')
        assert manifest['status'] == 'passed'
        assert manifest['source_run_report_sha256'] == digest(PREVIOUS / name / 'report.json')
        assert manifest['source_delta_sha256'] == source['delta']['sha256']
        assert manifest['source_delta_bytes'] == source['delta']['bytes']
        assert manifest['prepared_sha256'] == source['prepared_sha256']
        assert manifest['source_vocabulary_sha256'] == source['identity']['vocabulary_sha256']
        assert manifest['head_sizes'] == source['identity']['head_sizes']
        assert manifest['shape'] == [sum(manifest['head_sizes']), 32]
        assert manifest['row_bytes'] == sum(manifest['head_sizes']) * 128
        assert manifest['source_editing_metrics'] == source['metrics']
        assert manifest['unaddressed_budget_padding_rows'] == source['physical_rows'] - source['identity']['logical_exact_rows_with_zero_sentinels']
        assert len(manifest['cases']) == 64
        variants = {v['name']: v for v in manifest['variants']}
        assert set(variants) == {'identity', 'xor', 'dedup-xor', 'datrie-layout', 'datrie-exact'}
        queries = {}
        for mode in ('dense', 'raw-mmap', *variants):
            q = load(folder / f'query-{mode}.json')
            assert q['status'] == 'passed' and q['all_query_outputs_byte_exact']
            assert q['manifest_sha256'] == digest(folder / 'manifest.json')
            assert q['environment']['source_sha256'] == digest(root / 'runtime' / q['environment']['commit'] / 'tools/benchmark_counterfact_storage.py')
            assert len(q['samples_ms']) == len(q['cpu_samples_ms']) == 64 * q['repeat']
            assert len(q['query_row_counts']) == 64
            close(q['p50_ms'], statistics.median(q['samples_ms']))
            close(q['p95_ms'], percentile(q['samples_ms'], 95))
            queries[mode] = {'p50_ms': q['p50_ms'], 'p95_ms': q['p95_ms'],
                            'rss_after_open_bytes': q['rss_after_open']['current_bytes'],
                            'rss_after_query_bytes': q['rss_after_query']['current_bytes']}
        for v in variants.values():
            assert v['all_original_rows_byte_exact']
            close(v['row_payload_to_memory_artifact_factor'], manifest['row_bytes'] / v['memory_artifact_bytes'])
            close(v['source_bundle_to_native_bundle_factor'], manifest['source_identity_bundle_bytes'] / v['bundle_bytes'])
        assert variants['datrie-layout']['structurally_merged_rows'] == 0
        models = {}
        reference = load(folder / 'model-source.json')
        labels = ['source', 'xor'] + (['datrie-layout'] if name.endswith('-s3') else [])
        for label in labels:
            m = load(folder / f'model-{label}.json')
            assert m['status'] == 'passed' and m['execution_device'] == 'cuda'
            assert m['source_run_report_sha256'] == manifest['source_run_report_sha256']
            # These runs' environment() is the shared reporting helper, whose
            # __file__ is the storage benchmark. Both modules are archived at
            # the reported full commit; do not relabel a recorded digest.
            assert m['environment']['source_sha256'] == digest(root / 'runtime' / m['environment']['commit'] / 'tools/benchmark_counterfact_storage.py')
            assert m['source_delta_sha256'] == manifest['source_delta_sha256']
            assert m['source_row_bytes'] == manifest['row_bytes']
            assert m['memory_row_parameter_present'] == (label == 'source')
            assert len(m['requests']) == 64 and m['requests'] == reference['requests']
            for original, row_query in zip(m['requests'], manifest['cases']):
                assert (original['case_id'], original['group'], original['input_tokens'], original['input_ids_sha256']) == (row_query['case_id'], row_query['group'], row_query['tokens'], row_query['input_ids_sha256'])
            assert len(m['samples_ms']) == 64 * m['repeat']
            close(m['p50_ms'], statistics.median(m['samples_ms']))
            close(m['p95_ms'], percentile(m['samples_ms'], 95))
            close(m['requests_per_second'], len(m['samples_ms'])/(sum(m['samples_ms'])/1000))
            for point in ('gpu_before', 'gpu_before_timing', 'gpu_after_timing'):
                assert not m[point]['external_compute_pids'], point
            models[label] = {key: m[key] for key in ('open_seconds', 'p50_ms', 'p95_ms', 'requests_per_second',
                                                    'rss_after_open', 'rss_after_query', 'cuda_after_open', 'cuda_after_query')}
            if label != 'source':
                pairs += 64
        model_cost = models['source']['cuda_after_open']['allocated_bytes'] - models['xor']['cuda_after_open']['allocated_bytes']
        runs.append({'run': name, 'row_bytes': manifest['row_bytes'], 'source_bundle_bytes': manifest['source_identity_bundle_bytes'],
            'gzip_bytes': manifest['gzip']['bytes'], 'variants': variants, 'cpu_queries': queries,
            'model_costs': models, 'model_startup_cuda_allocated_reduction_bytes': model_cost,
            'native_over_source_p50_ratio': models['xor']['p50_ms'] / models['source']['p50_ms']})
    sweeps = []
    for name in sorted(f'{m}-20877-exact-s3' for m in ('qwen06', 'qwen35')):
        folder = root / 'block-sweep' / name
        m = load(folder / 'manifest.json')
        assert m['status'] == 'passed'
        assert m['source_fixture_manifest_sha256'] == digest(root / 'results' / name / 'manifest.json')
        assert [v['block_rows'] for v in m['variants']] == [32, 128, 512, 2048]
        variants = []
        for v in m['variants']:
            q = load(folder / f"query-b{v['block_rows']}.json")
            assert v['all_original_rows_byte_exact'] and q['all_query_outputs_byte_exact'] and q['status'] == 'passed'
            assert q['manifest_sha256'] == digest(folder / 'manifest.json')
            close(v['row_payload_to_memory_artifact_factor'], m['row_bytes'] / v['memory_artifact_bytes'])
            close(q['p50_ms'], statistics.median(q['samples_ms']))
            variants.append({**v, 'p50_ms': q['p50_ms'], 'first_trace_p50_ms': statistics.median(q['first_trace_ms'])})
        sweeps.append({'run': name, 'variants': variants})
    rss_audits = []
    for name in sorted(f'{m}-20877-exact-s3' for m in ('qwen06', 'qwen35')):
        folder = root / 'rss-audit' / name
        queries = {}
        models = {}
        for label in ('dense', 'raw-mmap', 'xor', 'datrie-layout'):
            q = load(folder / f'query-{label}.json')
            assert q['status'] == 'passed' and q['all_query_outputs_byte_exact']
            assert q['manifest_sha256'] == digest(root / 'results' / name / 'manifest.json')
            queries[label] = {key: q[key] for key in ('rss_before_open', 'rss_after_open', 'rss_after_query')}
        source = load(folder / 'model-source.json')
        for label in ('source', 'native'):
            m = load(folder / f'model-{label}.json')
            assert m['status'] == 'passed' and len(m['requests']) == 3 and m['requests'] == source['requests']
            models[label] = {key: m[key] for key in ('rss_before_open', 'rss_after_open', 'rss_after_query')}
        for item in [*queries.values(), *models.values()]:
            for point in item.values():
                assert point['anonymous_bytes'] >= 0 and point['file_backed_bytes'] >= 0
        rss_audits.append({'run': name, 'cpu_query_rss': queries, 'model_rss': models,
                           'limitation': 'Separate seed-3 process-wide RSS audits; 3 model prefixes, 64 CPU row prefixes. No timing or cross-seed RSS inference.'})
    return {'status': 'verified', 'trained_checkpoints': len(runs), 'native_complete_prefix_pairs': pairs,
            'runs': runs, 'block_sweeps': sweeps, 'rss_audits': rss_audits}


def verify(root=ROOT):
    manifest = load(root / 'manifest.json')['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p != root / 'manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(manifest), 'archive file set changed'
    for name, info in manifest.items():
        assert (root / name).stat().st_size == info['bytes'] and digest(root / name) == info['sha256'], name
    result = summary(root)
    assert result == load(root / 'summary.json'), 'published summary arithmetic changed'
    assert result['native_complete_prefix_pairs'] == 512
    assert load(root / 'storage-status.json')['status'] == 'complete'
    assert load(root / 'model-status.json')['status'] == 'complete'
    assert load(root / 'sweep-status.json')['status'] == 'complete'
    assert load(root / 'rss-status.json')['status'] == 'complete'
    print('Verified 6 trained tables, 512 paired complete CUDA prefixes, all CPU queries and fixed block-size costs.')


if __name__ == '__main__':
    verify()
