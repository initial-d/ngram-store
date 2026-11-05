"""Frozen Python/native full-assessment comparison at unchanged error budgets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np

from .analyze_budgeted_bags import BUDGETS, read_report_bytes
from .budgeted_bags import BudgetedBagRows
from .external_cotraining import file_hash


ATTRS = ('vectors', 'projection_l2_bounds', 'rounding_l2_bounds',
         'absolute_l2_bounds', 'relative_l2_bounds', 'exact_domain')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--prior-retiming', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--traces-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: CUDA_VISIBLE_DEVICES must be empty')
    if cfg.output.exists():
        raise FileExistsError(cfg.output)
    raw_bytes = read_report_bytes(cfg.report)
    old = json.loads(raw_bytes)
    prior = json.loads(cfg.prior_retiming.read_text())
    if old['status'] != 'passed' or prior['status'] != 'passed' or len(old['runs']) != 18:
        raise ValueError('complete passed previous experiments required')
    if prior['frozen_report_sha256'] != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError('previous retiming and original report differ')
    rng = np.random.default_rng(42)
    report = {'status': 'running', 'category': 'frozen trained-source CPU native bounded-plan implementation follow-up',
        'source_commit': cfg.source_commit, 'command': shlex.join([sys.executable, *sys.argv]),
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'cpu': json.loads(subprocess.check_output(['lscpu', '--json'], text=True)),
        'compiler': subprocess.check_output(['cc', '--version'], text=True),
        'native_library_sha256': file_hash(os.environ['MEFT_ROWS_LIBRARY']),
        'source_sha256': {name: file_hash(Path(__file__).resolve().parents[1] / name) for name in
            ('native/meft_rows.c', 'native/meft_rows.h', 'tools/budgeted_bags.py', 'tools/native_row_store.py')},
        'frozen_report_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'prior_retiming_sha256': file_hash(cfg.prior_retiming),
        'threads': 1, 'warmups': 2, 'repeat': 10, 'seed': 42, 'budgets': BUDGETS, 'runs': [],
        'limitations': ['Existing finite SGNS corpus/checkpoints, no new training or semantic/LLM quality.',
            'Same files/weights/error budgets; native compiler does not add storage or coverage savings.',
            'Full warm assessment including refused candidates; startup/hashing outside query timing.',
            'Source FP64 controls retain all source rows; candidate natives/sidecars retain compressed memory.',
            'Shared CPU, no exclusive/cold-cache/RSS, GPU, energy or monetary result.']}
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for run in old['runs']:
        name = f"order-{run['order']}-seed-{run['seed']}"
        root = cfg.results_root / run['convention'] / name
        if file_hash(root / 'source.f32') != run['source_sha256']:
            raise ValueError('trained FP32 source changed')
        source = np.memmap(root / 'source.f32', mode='r', dtype='<f4', shape=tuple(run['build']['shape']))
        traces = json.loads((cfg.traces_root / name / 'common/trace-bags.json').read_text())
        reference = next(r for r in prior['runs'] if (r['convention'], r['order'], r['seed']) ==
                         (run['convention'], run['order'], run['seed']))
        record = {'convention': run['convention'], 'order': run['order'], 'seed': run['seed'],
            'source_sha256': run['source_sha256'], 'artifact_sha256': run['build']['artifact_sha256'],
            'sidecar_files_sha256': run['bounds_build']['files_sha256'],
            'complete_model_bytes_unchanged': run['capacity']['budgeted_complete_model_bytes'], 'traces': {}}
        with BudgetedBagRows(root / 'store', root / 'bounds', compiler='python') as python, \
             BudgetedBagRows(root / 'store', root / 'bounds', compiler='native') as native:
            if file_hash(root / 'store/table.men') != record['artifact_sha256']:
                raise ValueError('native table changed')
            for file, digest in record['sidecar_files_sha256'].items():
                if file_hash(root / 'bounds' / file) != digest:
                    raise ValueError('sidecar changed')
            for kind, trace in traces.items():
                bags = trace[0]
                digest = hashlib.sha256(json.dumps(bags, separators=(',', ':')).encode()).hexdigest()
                if digest != run['trace_sha256'][kind]:
                    raise ValueError('query trace changed')
                for mean in (False, True):
                    a, b = python.assess(bags, mean=mean), native.assess(bags, mean=mean)
                    if any(getattr(a, key).tobytes() != getattr(b, key).tobytes() for key in ATTRS):
                        raise AssertionError('native compiler changed result/bound/domain bytes')
                    for budget in BUDGETS:
                        if a.accepts(max_relative_l2=budget).tobytes() != b.accepts(max_relative_l2=budget).tobytes():
                            raise AssertionError('native compiler changed an error-budget decision')
                if hashlib.sha256(b.vectors.tobytes()).hexdigest() != reference['traces'][kind]['output_sha256']:
                    raise AssertionError('mean output differs from the published frozen reference')
                if hashlib.sha256(b.absolute_l2_bounds.tobytes()).hexdigest() != reference['traces'][kind]['absolute_bounds_sha256']:
                    raise AssertionError('mean bound differs from the published frozen reference')
                def source_query():
                    return [source[bag].astype(np.float64).mean(axis=0) if bag else np.zeros(source.shape[1]) for bag in bags]
                for _ in range(2):
                    python.assess(bags, mean=True); native.assess(bags, mean=True); source_query()
                item = {'trace_sha256': digest, 'queries': len(bags), 'sum_mean_all_array_bytes_identical': True,
                    'all_budget_decisions_identical': True,
                    'mean_output_sha256': hashlib.sha256(b.vectors.tobytes()).hexdigest(),
                    'mean_absolute_bounds_sha256': hashlib.sha256(b.absolute_l2_bounds.tobytes()).hexdigest(),
                    'python_seconds': [], 'native_seconds': [], 'source_fp64_seconds': [], 'order': []}
                for _ in range(10):
                    first = int(rng.integers(2))
                    item['order'].append('python-first' if first == 0 else 'native-first')
                    for index in (first, 1-first):
                        before = time.perf_counter()
                        (python if index == 0 else native).assess(bags, mean=True)
                        item['python_seconds' if index == 0 else 'native_seconds'].append(time.perf_counter()-before)
                    before = time.perf_counter(); source_query()
                    item['source_fp64_seconds'].append(time.perf_counter()-before)
                for backend in ('python', 'native', 'source_fp64'):
                    item[backend + '_p50_seconds'] = float(np.median(item[backend + '_seconds']))
                item['improvement'] = item['python_p50_seconds'] / item['native_p50_seconds']
                record['traces'][kind] = item
        source._mmap.close()
        report['runs'].append(record)
        cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'completed': [run['convention'], run['order'], run['seed']],
            'oov_native_ms': record['traces']['oov']['native_p50_seconds']*1000,
            'oov_improvement': record['traces']['oov']['improvement']}), flush=True)
    report['status'] = 'passed'
    report['seconds'] = time.perf_counter()-started
    cfg.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
