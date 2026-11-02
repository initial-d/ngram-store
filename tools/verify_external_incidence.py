"""Separate, resident-runtime verification of an external incidence build.

This process deliberately materializes maps and native state. Its RSS must not
be included in the external construction result. Inputs are trusted artifacts.
"""
import argparse
import hashlib
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

from .cotraining_groups import BagPartition, CertifiedBagPlan, CertifiedBagRows
from .external_cotraining import ExternalBagPartition, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--previous-store', type=Path, required=True)
    p.add_argument('--bags', type=Path, required=True)
    p.add_argument('--expected-trace-sha256', required=True)
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    if cfg.output.exists():
        raise FileExistsError(cfg.output)
    started = time.perf_counter()
    bags = json.loads(cfg.bags.read_text())
    flat = np.asarray([row for bag in bags for row in bag], dtype='<u8')
    offsets = np.asarray([0, *np.cumsum([len(b) for b in bags])], dtype='<u8')
    trace = hashlib.sha256(flat.tobytes() + offsets.tobytes()).hexdigest()
    if trace != cfg.expected_trace_sha256:
        raise ValueError('native query trace changed')
    external = ExternalBagPartition(cfg.build / 'partition')
    partition = external.materialize()
    loaded = BagPartition.load(cfg.build / 'domain.npz')
    if loaded.summary() != partition.summary() or loaded.mapping.tobytes() != partition.mapping.tobytes():
        raise ValueError('exported legacy reader differs from materialized map')
    plan = CertifiedBagPlan(partition, bags)
    with CertifiedBagRows(cfg.previous_store, cfg.native_library) as previous:
        old_plan = previous.compile(bags)
        if previous.partition.summary() != partition.summary() or \
                previous.partition.mapping.tobytes() != partition.mapping.tobytes() or \
                previous.partition.sizes.tobytes() != partition.sizes.tobytes():
            raise ValueError('partition changed from the previously published native certificate')
        arrays = {}
        for name in ('ids', 'weights', 'offsets', 'denominators'):
            left, right = getattr(plan, name), getattr(old_plan, name)
            if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                raise ValueError('native plan bytes changed: ' + name)
            arrays[name] = {'bytes': left.nbytes, 'sha256': hashlib.sha256(left.tobytes()).hexdigest()}
        if plan.summary() != old_plan.summary():
            raise ValueError('native plan summary changed')
        # Check all new group means against the earlier native canonical rows,
        # without allocating a second complete float table.
        representatives = partition.representatives()
        checked = 0
        with (cfg.build / 'means.f32').open('rb') as means:
            for begin in range(0, len(representatives), 4096):
                ids = representatives[begin:begin + 4096]
                actual = previous.table.rows(ids)
                expected = means.read(len(ids) * previous.dim * 4)
                if actual != expected:
                    raise ValueError('new means differ from earlier native canonical rows')
                checked += len(ids)
            if means.read(1):
                raise ValueError('unexpected extra group means')
        outputs = {}
        for mean in (False, True):
            old, new = previous.evaluate(old_plan, mean=mean), previous.evaluate(plan, mean=mean)
            if old.tobytes() != new.tobytes():
                raise ValueError('same native reducer output bytes changed')
            outputs['mean' if mean else 'sum'] = {
                'bytes': new.nbytes, 'sha256': hashlib.sha256(new.tobytes()).hexdigest()}
    report = {
        'status': 'completed', 'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'command': shlex.join([sys.executable, *sys.argv]), 'device': 'cpu',
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'source_files_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
            ('verify_external_incidence.py', 'external_cotraining.py', 'cotraining_groups.py')},
        'native_library_sha256': file_hash(cfg.native_library),
        'previous_manifest_sha256': file_hash(cfg.previous_store / 'manifest.json'),
        'previous_domain_sha256': file_hash(cfg.previous_store / 'domain.npz'),
        'new_external_metadata_sha256': file_hash(cfg.build / 'partition/external.json'),
        'new_domain_file_sha256': file_hash(cfg.build / 'domain.npz'),
        'means_sha256': file_hash(cfg.build / 'means.f32'), 'bags_file_sha256': file_hash(cfg.bags),
        'query_bags_sha256': trace, 'partition': partition.summary(),
        'partition_arrays_and_legacy_reader_identical': True,
        'all_group_means_identical_to_previous_native_rows': checked,
        'plan': plan.summary(), 'identical_plan_arrays': arrays,
        'identical_native_outputs': outputs, 'seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1 if sys.platform == 'darwin' else 1024),
        'limitations': ['Separate resident verifier; maps and native runtime are O(N).',
            'Same existing native artifact and C reducer; no new native packing benchmark.',
            'Byte identity is to the existing projected means/plan, not the original source rows.',
            'No query-performance, new training or LLM-quality claim.']}
    with cfg.output.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'verified': plan.summary(), 'peak_rss_bytes': report['peak_rss_bytes']}))


if __name__ == '__main__':
    main()
