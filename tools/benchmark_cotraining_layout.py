"""Lossless control: group actual source rows by training incidence, never average.

Independent from aggregate-certified canonicalization. Original numeric IDs and
all original FP32 bytes are verified under identity and incidence layouts. A
training signature may improve locality but does not itself establish a gain.
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
import torch

from .benchmark_cotraining import fit_partition, write_json
from .cotraining_groups import BagPartition
from .engram_row_adapter import file_sha256
from .native_row_store import NativeRows, pack_native


def run(source, partition, dim, common_bytes, library, directory, source_manifest, seed):
    directory.mkdir(parents=True, exist_ok=False)
    shape = (len(partition.mapping), dim)
    rows = np.memmap(source, mode='r', dtype='<f4', shape=shape)
    rng = np.random.default_rng(seed)
    # Identical traces across layouts; actual source IDs, not fabricated rows.
    traces = {'uniform': rng.integers(len(rows), size=512).tolist(),
              'hot': rng.choice(min(128, len(rows)), size=512).tolist()}
    variants = []
    for name, order in [('identity', None), ('incidence-layout', np.argsort(partition.mapping, kind='stable').tolist())]:
        artifact = directory / (name + '.men')
        started = time.perf_counter()
        packed = pack_native(source, artifact, dim * 4, block_rows=max(1, 8192 // (dim * 4)),
                             deduplicate=False, row_order=order, predictor='xor')
        digest = hashlib.sha256()
        with NativeRows(artifact, library) as native:
            for begin in range(0, len(rows), 4096):
                end = min(begin + 4096, len(rows))
                actual = native.rows(range(begin, end))
                if actual != rows[begin:end].tobytes():
                    raise AssertionError('source row bytes/IDs changed in lossless layout')
                digest.update(actual)
        if digest.hexdigest() != source_manifest['source_sha256']:
            raise AssertionError('source digest mismatch')
        variant = {'name': name, 'storage': packed, 'artifact_sha256': file_sha256(artifact),
                   'all_source_rows_byte_exact': len(rows), 'decoded_sha256': digest.hexdigest(),
                   'build_and_verification_seconds': time.perf_counter() - started,
                   'complete_bytes': packed['artifact_bytes'] + common_bytes, 'query': {}}
        # Fresh handle per trace. OS pages remain warm after verification.
        for trace_name, ids in traces.items():
            samples = []
            with NativeRows(artifact, library, cache_bytes=4 << 20) as native:
                for _ in range(9):
                    start = time.perf_counter()
                    actual = native.rows(ids)
                    samples.append(time.perf_counter() - start)
                    if actual != rows[ids].tobytes():
                        raise AssertionError('query row byte mismatch')
            variant['query'][trace_name] = {'samples_seconds': samples,
                                            'ids_sha256': hashlib.sha256(np.asarray(ids, dtype='<u8').tobytes()).hexdigest()}
        variants.append(variant)
    return {'shape': list(shape), 'source': source_manifest, 'partition': partition.summary(),
            'common_bytes': common_bytes, 'variants': variants,
            'extra_complete_saving': 1 - variants[1]['complete_bytes'] / variants[0]['complete_bytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--training-dir', type=Path)
    parser.add_argument('--partition-path', type=Path)
    parser.add_argument('--max-orders', default='4,8,12')
    parser.add_argument('--seeds', default='3,7,11')
    cfg = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or bool(cfg.checkpoint_root) == bool(cfg.training_dir):
        parser.error('CPU-only empty CUDA_VISIBLE_DEVICES; choose exactly one source')
    if cfg.training_dir and not cfg.partition_path:
        parser.error('large source needs its verified incidence partition')
    torch.set_num_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
              'numpy': np.__version__, 'torch': torch.__version__, 'os': platform.platform(),
              'device': 'cpu', 'threads': 1,
              'limitations': ['Lossless layout of actual source rows, not aggregate projection or newly trained memory.',
                              'One finite SGNS family; source provenance retained, no convergence/LLM task claim.',
                              'Signatures only guide export order; no certificate is needed at lookup, original bytes restored.',
                              'Query samples follow byte verification with warm OS pages on shared CPU; not exclusive/cold latency.',
                              'Small source complete bytes include input table + output vectors + common keys; large source excludes unavailable word/output model.',
                              'No GPU touched; monetary/energy cost unmeasured.'], 'runs': []}
    write_json(cfg.output_dir / 'protocol.json', report)
    if cfg.training_dir:
        training = json.loads((cfg.training_dir / 'training.json').read_text())
        source = cfg.training_dir / 'trained.f32'
        if file_sha256(source) != training['trained_sha256']:
            raise ValueError('trained source hash mismatch')
        partition = BagPartition.load(cfg.partition_path)
        common = sum((cfg.training_dir / name).stat().st_size for name in ('keys.json', 'trained-source-ids.u64'))
        report['runs'].append(run(source, partition, training['dim'], common, cfg.native_library,
            cfg.output_dir / 'large', {'source_sha256': training['trained_sha256'],
                                     'partition_file_sha256': file_sha256(cfg.partition_path)}, 42))
    else:
        for order in map(int, cfg.max_orders.split(',')):
            for seed in map(int, cfg.seeds.split(',')):
                checkpoint_path = cfg.checkpoint_root / f'order-{order}-seed-{seed}/reference.pt'
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                rows = checkpoint['input_rows'].numpy().astype('<f4')
                partition = fit_partition(checkpoint['bags'], len(rows), 1 + len(checkpoint['words']))
                source = cfg.output_dir / f'staging-{order}-{seed}.f32'
                rows.tofile(source)
                common = checkpoint['output_rows'].numel() * 4 + len(json.dumps([checkpoint['words'], checkpoint['grams']]).encode())
                result = run(source, partition, rows.shape[1], common, cfg.native_library,
                    cfg.output_dir / f'order-{order}-seed-{seed}', {'source_sha256': file_sha256(source),
                    'checkpoint_sha256': file_sha256(checkpoint_path), 'order': order, 'seed': seed}, seed)
                source.unlink()
                report['runs'].append(result)
                write_json(cfg.output_dir / 'report.json', report)
    report['status'] = 'completed'
    report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(cfg.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
