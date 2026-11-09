"""Check archive integrity and paired format arithmetic using only the stdlib.

Raw trained rows remain in the original source runs. This verifier audits
published provenance, output-hash assertions and timing/RSS arithmetic; rerun
the benchmark with original rows to independently reconstruct the vectors.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import shlex

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'qwen_native_identity_evidence_20261109/results'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def percentile(samples, p):
    s = sorted(samples)
    position = (len(s) - 1) * p / 100
    lo = int(position)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (position - lo)


def summary(root=ROOT):
    expected = {f'{model}-20877-exact-s{seed}' for model in ('qwen06', 'qwen35') for seed in (3, 7, 11)}
    assert {p.name for p in (root / 'results').iterdir()} == expected
    build = load(root / 'build.json')
    runs = []
    workers = pairs = 0
    for name in sorted(expected):
        folder = root / 'results' / name
        previous = load(PREVIOUS / name / 'manifest.json')
        manifest = load(folder / 'manifest.json')
        assert manifest['status'] == 'passed'
        assert manifest['environment']['commit'] == build['source_commit']
        snapshot = root / 'runtime' / build['source_commit']
        assert manifest['environment']['source_sha256'] == digest(snapshot / 'tools/benchmark_counterfact_storage.py')
        assert manifest['source_fixture_manifest_sha256'] == digest(PREVIOUS / name / 'manifest.json')
        assert manifest['old_library_sha256'] == previous['native_library_sha256']
        assert manifest['new_library_sha256'] != manifest['old_library_sha256']
        for field in ('source_delta_sha256', 'source_row_sha256', 'queries_sha256', 'cases', 'shape',
                      'head_sizes', 'offsets', 'row_bytes', 'source_editing_metrics'):
            assert manifest[field] == previous[field], field
        nrows, dim = manifest['shape']
        assert dim == 32 and manifest['row_bytes'] == nrows * 128
        blocks = [32, 128, 512, 2048] if name.endswith('-s3') else [32]
        variants = {v['name']: v for v in manifest['variants']}
        assert set(variants) == {f'v{v}-b{b}' for v in (2, 3) for b in blocks}
        assert len(manifest['paired_layouts']) == len(blocks)
        for item, block in zip(manifest['paired_layouts'], blocks):
            assert item == {'block_rows': block, 'saved_bytes': nrows * 8,
                            'identical_codec_payload_and_descriptors': True}
        rows = []
        schedule = load(folder / 'schedule.json')
        assert len(schedule) == 6 * len(blocks)
        assert load(folder / 'complete.json') == {'status': 'complete', 'workers': len(schedule)}
        for block in blocks:
            old, new = [variants[f'v{v}-b{block}'] for v in (2, 3)]
            a, b = old['packed'], new['packed']
            assert a['format_version'] == 2 and b['format_version'] == 3
            assert a['row_map_bytes'] == nrows * 8 and b['row_map_bytes'] == 0
            assert a['layout'] == b['layout']
            assert a['layout']['header'] == [nrows, 128, nrows, block, math.ceil(nrows / block)]
            assert a['artifact_bytes'] - b['artifact_bytes'] == nrows * 8
            for v, variant in ((2, old), (3, new)):
                assert variant['all_original_rows_byte_exact']
                assert variant['packed']['reconstructed_sha256'] == manifest['source_row_sha256']
                meta_path = folder / variant['name'] / 'embedding.json'
                meta = load(meta_path)
                assert digest(meta_path) == variant['memory_manifest_sha256']
                assert meta['artifact_sha256'] == variant['table_sha256']
                assert meta['format_version'] == v
                assert variant['memory_artifact_bytes'] == variant['packed']['artifact_bytes'] + meta_path.stat().st_size
            assert a['old_reader_reconstructed_sha256'] == manifest['source_row_sha256']
            assert b['old_reader_rejection'] == 'invalid native row header'
            if block == 32:
                assert a['matches_previous_v2_artifact']
                previous_v2 = next(v for v in previous['variants'] if v['name'] == 'xor')
                assert old['table_sha256'] == previous_v2['table_sha256']
            query = {}
            for cache in (0, 1048576):
                for label, version in (('old-v2', 2), ('new-v2', 2), ('new-v3', 3)):
                    basename = f'{label}-b{block}-c{cache}'
                    q = load(folder / (basename + '.json'))
                    assert q['status'] == 'passed' and q['all_query_outputs_byte_exact']
                    assert q['manifest_sha256'] == digest(folder / 'manifest.json')
                    assert q['mode'] == f'v{version}-b{block}' and q['cache_bytes'] == cache
                    assert q['environment']['commit'] == build['source_commit']
                    assert q['environment']['source_sha256'] == digest(snapshot / 'tools/benchmark_counterfact_storage.py')
                    assert q['environment']['threads'] == q['environment']['interop_threads'] == 1
                    assert q['environment']['device'] == 'cpu' and q['environment']['gpu'] == 'unused'
                    assert q['repeat'] == 20 and q['warmup'] == 3
                    assert len(q['samples_ms']) == len(q['cpu_samples_ms']) == 64 * 20
                    assert len(q['first_trace_ms']) == len(q['query_row_counts']) == 64
                    close(q['p50_ms'], statistics.median(q['samples_ms']))
                    close(q['p95_ms'], percentile(q['samples_ms'], 95))
                    command = next(s for s in schedule if s['name'] == basename)['command']
                    invoked = shlex.split(q['environment']['command'])
                    # Python -m replaces sys.argv[0] with the resolved script;
                    # the schedule preserves the original module invocation.
                    assert invoked[0] == command[0] and invoked[2:] == command[3:]
                    assert invoked[1].endswith('/tools/benchmark_counterfact_storage.py')
                    assert command[1:4] == [
                        '-m', 'tools.benchmark_counterfact_storage', 'worker']
                    query[f'{label}-c{cache}'] = {
                        'p50_ms': q['p50_ms'], 'p95_ms': q['p95_ms'],
                        'first_trace_p50_ms': statistics.median(q['first_trace_ms']),
                        'cpu_p50_ms': statistics.median(q['cpu_samples_ms']),
                        'open_seconds': q['open_seconds'],
                        'rss_anonymous_growth_bytes': q['rss_after_query']['anonymous_bytes'] - q['rss_before_open']['anonymous_bytes'],
                        'rss_file_growth_bytes': q['rss_after_query']['file_backed_bytes'] - q['rss_before_open']['file_backed_bytes']}
                    workers += 1
            rows.append({'block_rows': block, 'v2_table_bytes': a['artifact_bytes'],
                         'v3_table_bytes': b['artifact_bytes'], 'saved_bytes': nrows * 8,
                         'v3_memory_artifact_bytes': new['memory_artifact_bytes'],
                         'row_payload_to_v3_memory_factor': manifest['row_bytes'] / new['memory_artifact_bytes'],
                         'queries': query})
            pairs += 1
        controls = {}
        if name.endswith('-s3'):
            for mode in ('dense', 'raw-mmap'):
                q = load(folder / (mode + '.json'))
                assert q['status'] == 'passed' and q['all_query_outputs_byte_exact']
                assert q['manifest_sha256'] == digest(folder / 'manifest.json')
                assert q['environment']['commit'] == build['source_commit']
                close(q['p50_ms'], statistics.median(q['samples_ms']))
                controls[mode] = {key: q[key] for key in ('p50_ms', 'p95_ms', 'open_seconds')}
                workers += 1
        runs.append({'run': name, 'physical_rows': nrows, 'row_payload_bytes': manifest['row_bytes'],
                     'layouts': rows, 'controls': controls})
    return {'status': 'verified', 'trained_checkpoints': len(runs), 'paired_layouts': pairs,
            'fresh_cpu_workers': workers, 'new_gpu_or_model_measurements': 0, 'runs': runs}


def verify(root=ROOT):
    files = load(root / 'manifest.json')['files']
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p != root / 'manifest.json' and '__pycache__' not in p.parts}
    assert actual == set(files)
    for name, info in files.items():
        path = root / name
        assert path.stat().st_size == info['bytes'] and digest(path) == info['sha256'], name
    result = summary(root)
    assert result == load(root / 'summary.json')
    assert result['paired_layouts'] == 12 and result['fresh_cpu_workers'] == 76
    print('Verified 6 trained tables, 12 matched v2/v3 layouts and 76 fresh CPU workers; no new GPU claim.')


if __name__ == '__main__':
    verify()
