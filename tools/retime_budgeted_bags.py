"""Paired same-source retiming of the first and batched error-bound paths."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np

from .budgeted_bags import BudgetedBagRows
from .external_cotraining import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--previous-traces-root', type=Path, required=True)
    p.add_argument('--reference-module', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only run requires CUDA_VISIBLE_DEVICES to be empty')
    raw = json.loads(cfg.report.read_text())
    if raw['status'] != 'passed' or len(raw['runs']) != 18 or raw['failures']:
        raise ValueError('complete passed frozen run required')
    if cfg.output.exists():
        raise FileExistsError(cfg.output)
    spec = importlib.util.spec_from_file_location('tools.budgeted_bags_reference', cfg.reference_module)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(42)
    report = {'status': 'running', 'source_commit': cfg.source_commit, 'command': shlex.join([sys.executable, *sys.argv]),
        'frozen_report_sha256': file_hash(cfg.report), 'reference_module_sha256': file_hash(cfg.reference_module),
        'module_sha256': file_hash(Path(__file__).with_name('budgeted_bags.py')),
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'warmups': 2, 'repeat': 5, 'seed': 42, 'runs': [],
        'limitations': ['Same already-trained sources/traces and native C engine, shared warm single-thread CPU.',
            'Full candidate assessment includes refused queries, rounding and residual bounds.',
            'Paired timing applies to nine legacy primary checkpoints; average sensitivity controls verify equality only.',
            'Faster than the first bounded path does not imply faster than dense or strict-subset queries.']}
    if report['reference_module_sha256'] != raw['source_files_sha256']['budgeted_bags.py']:
        raise ValueError('reference module differs from the frozen benchmark source')
    for run in raw['runs']:
        name = f"order-{run['order']}-seed-{run['seed']}"
        root = cfg.results_root / run['convention'] / name
        traces = json.loads((cfg.previous_traces_root / name / 'common/trace-bags.json').read_text())
        result = {'convention': run['convention'], 'order': run['order'], 'seed': run['seed'], 'traces': {}}
        with reference.BudgetedBagRows(root / 'store', root / 'bounds') as old, BudgetedBagRows(root / 'store', root / 'bounds') as new:
            for kind, trace in traces.items():
                bags = trace[0]
                digest = hashlib.sha256(json.dumps(bags, separators=(',', ':')).encode()).hexdigest()
                if digest != run['trace_sha256'][kind]:
                    raise ValueError('trace differs from the frozen source')
                a, b = old.assess(bags, mean=True), new.assess(bags, mean=True)
                attributes = ('vectors', 'projection_l2_bounds', 'rounding_l2_bounds',
                              'absolute_l2_bounds', 'relative_l2_bounds', 'exact_domain')
                if any(getattr(a, attr).tobytes() != getattr(b, attr).tobytes() for attr in attributes):
                    raise AssertionError('batching changed output, bound or exact-domain bytes')
                expected = run['traces'][kind]
                if b.absolute_l2_bounds.tolist() != expected['absolute_l2_bounds'] or [float(v) if np.isfinite(v) else None
                        for v in b.relative_l2_bounds] != expected['relative_l2_bounds']:
                    raise AssertionError('bounds differ from the frozen numerical report')
                item = {'trace_sha256': digest, 'all_output_bound_domain_bytes_identical': True,
                    'output_sha256': hashlib.sha256(b.vectors.tobytes()).hexdigest(),
                    'absolute_bounds_sha256': hashlib.sha256(b.absolute_l2_bounds.tobytes()).hexdigest()}
                if run['convention'] == 'legacy':
                    for _ in range(2):
                        old.assess(bags, mean=True)
                        new.assess(bags, mean=True)
                    samples = {'reference_seconds': [], 'batched_seconds': [], 'order': []}
                    for _ in range(5):
                        first = int(rng.integers(2))
                        samples['order'].append('reference-first' if first == 0 else 'batched-first')
                        for index in (first, 1-first):
                            before = time.perf_counter()
                            (old if index == 0 else new).assess(bags, mean=True)
                            samples['reference_seconds' if index == 0 else 'batched_seconds'].append(time.perf_counter()-before)
                    item.update(samples)
                    item['reference_p50_seconds'] = float(np.median(samples['reference_seconds']))
                    item['batched_p50_seconds'] = float(np.median(samples['batched_seconds']))
                    item['improvement'] = item['reference_p50_seconds'] / item['batched_p50_seconds']
                result['traces'][kind] = item
        report['runs'].append(result)
        cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'completed': [run['convention'], run['order'], run['seed']],
            'oov_improvement': result['traces']['oov'].get('improvement')}), flush=True)
    report['status'] = 'passed'
    cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
