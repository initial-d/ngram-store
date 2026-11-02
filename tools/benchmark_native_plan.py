"""Paired production native-plan compilation and optional native bag queries.

Consumes actual registered-ID traces; no fabricated embedding vectors. Source
row fidelity is outside this retiming: old/new plan and output bytes must agree.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

import numpy as np

from .benchmark_faiss_cotraining import digest_file, write_json
from .cotraining_groups import BagPartition, CertifiedBagPlan, CertifiedBagRows


def native_evaluate(native, plan):
    """The same C weighted reducer for old/new plans, followed by bag mean."""
    output = np.empty((len(plan.denominators), native.dim), dtype=np.float32)
    if len(output):
        native.table.embedding_bag_into(plan.ids.ctypes.data, len(plan.ids), plan.offsets.ctypes.data,
            len(output), 1, False, (1 << 64) - 1, 64, output.ctypes.data, output.size,
            weights_address=plan.weights.ctypes.data, weights_bytes=plan.weights.nbytes)
        output /= plan.denominators[:, None]
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partition', type=Path, required=True)
    p.add_argument('--bags', type=Path, required=True, help='JSON list of original-ID bags')
    p.add_argument('--reference-module', type=Path, required=True)
    p.add_argument('--native-store', type=Path)
    p.add_argument('--native-library', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=15)
    p.add_argument('--warmup', type=int, default=2)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or cfg.repeat < 1 or cfg.warmup < 1:
        p.error('CPU only with empty CUDA_VISIBLE_DEVICES and positive repeat/warmup')
    spec = importlib.util.spec_from_file_location('tools._reference_native_groups', cfg.reference_module)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    old_partition, partition = reference.BagPartition.load(cfg.partition), BagPartition.load(cfg.partition)
    if old_partition.summary() != partition.summary():
        raise ValueError('partition semantics changed')
    bags = json.loads(cfg.bags.read_text())
    def compile_old():
        return reference.CertifiedBagPlan(old_partition, bags)
    def compile_new():
        return CertifiedBagPlan(partition, bags)
    old, new = compile_old(), compile_new()
    for key in ('ids', 'weights', 'offsets', 'denominators'):
        if getattr(old, key).tobytes() != getattr(new, key).tobytes():
            raise ValueError('native plan bytes changed: ' + key)
    if old.summary() != new.summary():
        raise ValueError('native plan summary changed')
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_files_sha256': {name: digest_file(Path(__file__).with_name(name)) for name in
                             ('benchmark_native_plan.py', 'cotraining_groups.py')},
        'reference_module_sha256': digest_file(cfg.reference_module),
        'command': shlex.join([sys.executable, *sys.argv]),
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'cpu': subprocess.check_output(['lscpu', '-J'], text=True), 'device': 'cpu',
        'partition_file_sha256': digest_file(cfg.partition), 'bags_file_sha256': digest_file(cfg.bags),
        'partition': partition.summary(), 'plan': new.summary(), 'plan_arrays_byte_identical': True,
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
        'limitations': ['Actual registered trace, unchanged weighted reducer; no new model-quality claim.',
            'Shared warm CPU and bounded decoded cache, not exclusive or disk-cold latency.',
            'Compilation includes original-ID validation/counts, certification and plan construction.',
            'New partition has immutable buffers and cached digest; reference recomputes digest/representatives.',
            'No fallback, OOV, arbitrary original-row fidelity or GPU performance claim.',
            'RSS includes partition loading, validation and runtime; no isolated memory reduction claim.']}
    functions = {'compile_old': compile_old, 'compile_new': compile_new}
    native = None
    try:
        if cfg.native_store:
            native = CertifiedBagRows(cfg.native_store, cfg.native_library)
            if native.partition.summary() != partition.summary():
                raise ValueError('native store belongs to another partition')
            native.table.set_cache_bytes(4 << 20)
            a, b = native_evaluate(native, old), native.evaluate(new, mean=True)
            if a.tobytes() != b.tobytes():
                raise ValueError('native output bytes changed')
            report['native_table_sha256'] = digest_file(cfg.native_store / 'table.men')
            report['native_library_sha256'] = digest_file(cfg.native_library) if cfg.native_library else None
            report['native_outputs_byte_identical'] = True
            report['native_output_sha256'] = hashlib.sha256(a.tobytes()).hexdigest()
            functions.update(replay=lambda: native_evaluate(native, old),
                             dynamic_old=lambda: native_evaluate(native, compile_old()),
                             dynamic_new=lambda: native_evaluate(native, compile_new()))
        samples = {key: [] for key in functions}
        rng = np.random.default_rng(42)
        for _ in range(cfg.warmup):
            for function in functions.values():
                function()
        for _ in range(cfg.repeat):
            for key in rng.permutation(list(functions)):
                start = time.perf_counter()
                functions[key]()
                samples[key].append(time.perf_counter() - start)
        report['timing'] = {key: {'samples_seconds': values, 'p50_ms': float(np.median(values) * 1000),
                                  'p95_ms': float(np.quantile(values, .95) * 1000)} for key, values in samples.items()}
    finally:
        if native:
            native.close()
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    report['status'] = 'completed'
    write_json(cfg.output, report)


if __name__ == '__main__':
    main()
