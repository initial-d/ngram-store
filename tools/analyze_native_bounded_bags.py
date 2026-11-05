"""Validate and summarize the frozen CPU bounded-plan implementation follow-up."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from .analyze_budgeted_bags import BUDGETS, EXPECTED, read_report_bytes, validate


def summarize(report, original, prior):
    validate(original)
    if (report.get('status') != 'passed' or report.get('budgets') != BUDGETS
            or report.get('repeat') != 10 or report.get('warmups') != 2
            or report.get('threads') != 1 or report.get('seed') != 42):
        raise ValueError('passed predeclared CPU timing protocol required')
    keys = [(r['convention'], r['order'], r['seed']) for r in report['runs']]
    if len(keys) != 18 or set(keys) != EXPECTED:
        raise ValueError('18 unique frozen checkpoints required')
    if prior.get('status') != 'passed':
        raise ValueError('passed prior numerical reference required')
    originals = {(r['convention'], r['order'], r['seed']): r for r in original['runs']}
    references = {(r['convention'], r['order'], r['seed']): r for r in prior['runs']}
    for run in report['runs']:
        key = run['convention'], run['order'], run['seed']
        old, reference = originals[key], references[key]
        if (run['source_sha256'] != old['source_sha256']
                or run['artifact_sha256'] != old['build']['artifact_sha256']
                or run['sidecar_files_sha256'] != old['bounds_build']['files_sha256']
                or run['complete_model_bytes_unchanged'] != old['capacity']['budgeted_complete_model_bytes']
                or set(run['traces']) != {'known', 'oov'}):
            raise ValueError('source/table/sidecar/capacity/trace set changed')
        for kind, trace in run['traces'].items():
            if (not trace['sum_mean_all_array_bytes_identical'] or not trace['all_budget_decisions_identical']
                    or trace['trace_sha256'] != old['trace_sha256'][kind]
                    or trace['queries'] != old['traces'][kind]['queries']
                    or trace['mean_output_sha256'] != reference['traces'][kind]['output_sha256']
                    or trace['mean_absolute_bounds_sha256'] != reference['traces'][kind]['absolute_bounds_sha256']):
                raise ValueError('query/output/bound/budget contract changed')
            if len(trace['order']) != 10 or any(v not in ('python-first', 'native-first') for v in trace['order']):
                raise ValueError('ten paired randomized timing orders required')
            for backend in ('python', 'native', 'source_fp64'):
                samples = trace[backend + '_seconds']
                if len(samples) != 10 or any(not math.isfinite(v) or v <= 0 for v in samples):
                    raise ValueError('ten positive finite raw timing samples required')
                if trace[backend + '_p50_seconds'] != median(samples):
                    raise ValueError('timing median differs from raw samples')
            if trace['improvement'] != trace['python_p50_seconds'] / trace['native_p50_seconds']:
                raise ValueError('speedup denominator changed')
    result = {'source_commit': report['source_commit'],
        'all_18_sum_mean_output_bound_domain_bytes_and_budget_decisions_identical': True,
        'complete_model_bytes_unchanged': True, 'conventions': {},
        'scope': 'Frozen finite SGNS trained-source full CPU assessment; unchanged approximation contract, no new quality or compression claim.'}
    for convention in ('legacy', 'average'):
        rows = []
        for order in (4, 8, 12):
            selected = [r for r in report['runs'] if r['convention'] == convention and r['order'] == order]
            row = {'order': order, 'traces': {}}
            for kind in ('known', 'oov'):
                traces = [r['traces'][kind] for r in selected]
                row['traces'][kind] = {backend + '_ms': median(t[backend + '_p50_seconds'] * 1000 for t in traces)
                                      for backend in ('python', 'native', 'source_fp64')}
                row['traces'][kind]['paired_speedup'] = median(t['improvement'] for t in traces)
                row['traces'][kind]['native_vs_source_ratio'] = median(
                    t['native_p50_seconds'] / t['source_fp64_p50_seconds'] for t in traces)
            rows.append(row)
        result['conventions'][convention] = rows
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    cfg = p.parse_args()
    report = json.loads(cfg.report.read_text())
    original_bytes = read_report_bytes(cfg.original)
    prior_bytes = cfg.prior.read_bytes()
    if (report['frozen_report_sha256'] != hashlib.sha256(original_bytes).hexdigest()
            or report['prior_retiming_sha256'] != hashlib.sha256(prior_bytes).hexdigest()):
        raise ValueError('different frozen reports')
    summary = summarize(report, json.loads(original_bytes), json.loads(prior_bytes))
    cfg.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
