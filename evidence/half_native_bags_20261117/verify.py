"""Verify retained paired timings and inherited hash provenance, without NumPy."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(path):
    raw = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        raw = gzip.decompress(raw)
    return raw, json.loads(raw)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def main():
    _, summary = read(ROOT / 'summary.json')
    check(len(summary['cases']) == 10, 'complete initial/final five-case grid required')
    initial = {}
    for case in summary['cases']:
        raw, report = read(ROOT / case['report'])
        check(hashlib.sha256(raw).hexdigest() == case['report_sha256'], 'report hash changed')
        check(report['status'] == 'passed' and report['commit'] == summary[case['phase'] + '_commit'], 'status/source commit')
        check(report['rows'] == case['rows'] and report['stored_dtype'] == case['dtype'], 'source shape/dtype')
        check(report['row_bytes'] == 256 and report['dim'] == 128 and report['device'] == 'cpu', 'CPU half shape')
        check(report['chunk_rows'] == 4096 and report['mode'] == 'mean' and report['requests'] == 256, 'frozen query protocol')
        check(len(report['results']) == (1 if case['compressed'] else 2), 'complete batch grid')
        for cell, retained in zip(report['results'], case['results']):
            check(cell['batch_size'] == retained['batch_size'], 'batch summary')
            check(len(cell['round_results']) == (2 if case['compressed'] else 3), 'round count')
            ratios = []
            for round_result in cell['round_results']:
                samples = round_result['samples_us']; a = samples['baseline']; b = samples['candidate']
                check(len(a) == len(b) == report['repeat'] * math.ceil(256 / cell['batch_size']), 'timing count')
                check(all(math.isfinite(v) and v > 0 for v in a + b), 'invalid timing')
                ratio = math.exp(statistics.mean(math.log(x / y) for x, y in zip(a, b)))
                check(close(ratio, round_result['paired_geometric_speed_ratio']), 'paired ratio changed')
                ratios.append(ratio)
                check(round_result['source_fp64_to_fp32_features_byte_match'], 'feature mismatch')
                check(round_result['work']['baseline'] == round_result['work']['candidate'], 'logical work mismatch')
                for name in ('baseline', 'candidate'):
                    check(close(statistics.median(samples[name]), round_result['p50_us'][name]), 'p50 changed')
            ratio = math.exp(statistics.mean(math.log(r) for r in ratios))
            check(close(ratio, cell['geometric_speed_ratio']) and close(ratio, retained['speed_ratio']), 'summary ratio changed')
            for name in ('baseline', 'candidate'):
                check(close(statistics.median(r['p50_us'][name] for r in cell['round_results']), retained['p50_us'][name]), 'summary p50 changed')
        key = Path(case['report']).name
        if case['phase'] == 'initial':
            initial[key] = (raw, report)
        else:
            parent_raw, parent = initial[key]
            provenance = report['input_hash_provenance']
            check(provenance['parent_sha256'] == hashlib.sha256(parent_raw).hexdigest(), 'inherited parent changed')
            for name in ('table', 'codes', 'traces', 'trace_metadata', 'baseline_library'):
                check(report['hashes'][name] == parent['hashes'][name], 'frozen input hash changed: ' + name)
    initial_build_raw, initial_build = read(ROOT / 'initial-build.json')
    _, final_build = read(ROOT / 'final-build.json')
    check(final_build['initial_build_manifest_sha256'] == hashlib.sha256(initial_build_raw).hexdigest(), 'build provenance')
    check(len(initial_build['tables']) == 4, 'complete full table exports')
    for table in initial_build['tables']:
        rows = table['rows']; width = table['row_bytes']
        check(table['table_bytes'] == 56 + 32 * math.ceil(rows / 256) + rows * width, 'full identity layout size')
    for phase in ('initial', 'final'):
        _, status = read(ROOT / (phase + '-run-status.json'))
        check(len(status) == 5 and all(r['returncode'] == 0 for r in status), 'incomplete jobs')
    check('loop vectorized using 16 byte vectors' in (ROOT / 'tests/final-vectorization.log').read_text(), 'compiler diagnostics')
    print('Verified 18 timing cells, all paired samples, source-feature flags, complete table sizes and inherited hash provenance.')


if __name__ == '__main__':
    main()
