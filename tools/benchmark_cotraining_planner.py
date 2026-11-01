"""Isolate production DATrie mark/compact from canonicalization and byte dedup.

All three layouts use the SAME actual trained rows after the SAME certified
projection, codec and complete-cost accounting. Every logical row is verified.
Projection changes source vectors; it is not source-row lossless compression.
"""
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
import torch

from .benchmark_cotraining import fit_partition, write_json
from .engram_row_adapter import file_sha256
from .native_row_store import NativeRows, pack_native
from .structural_native import StructuralPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--structural-library', type=Path, required=True)
    cfg = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    torch.set_num_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
              'torch': torch.__version__, 'numpy': np.__version__, 'os': platform.platform(),
              'native_library_sha256': file_sha256(cfg.native_library),
              'structural_library_sha256': file_sha256(cfg.structural_library),
              'limitations': ['Nine existing finite SGNS checkpoints; no LLM/convergence claim.',
                              'All variants share a certified projection that changes source rows.',
                              'DATrie planning only; common JSON registry counted, no DATrie key-index storage/query claim.',
                              'Word/output rows, keys, manifests and certificates counted; schedules omitted equally.',
                              'CPU-only, shared host; no monetary/energy metering.'], 'runs': []}
    write_json(cfg.output_dir / 'protocol.json', report)
    for order in (4, 8, 12):
        for seed in (3, 7, 11):
            checkpoint_path = cfg.checkpoint_root / f'order-{order}-seed-{seed}/reference.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            rows, dim = checkpoint['input_rows'].numpy(), checkpoint['input_rows'].shape[1]
            offset = 1 + len(checkpoint['words'])
            partition = fit_partition(checkpoint['bags'], len(rows), offset)
            canonical = partition.means(rows)[partition.mapping].astype('<f4')
            keys = {offset + i: g for i, g in enumerate(checkpoint['grams'])}
            with StructuralPlan(canonical.tobytes(), dim, keys, cfg.structural_library, mode='exact') as plan:
                representatives = plan.representatives
            directory = cfg.output_dir / f'order-{order}-seed-{seed}'
            directory.mkdir()
            common = directory / 'common'
            common.mkdir()
            (common / 'keys.json').write_bytes(json.dumps([checkpoint['words'], checkpoint['grams']], ensure_ascii=True).encode())
            checkpoint['output_rows'].numpy().astype('<f4').tofile(common / 'output.f32')
            partition.save(common / 'domain.npz')
            common_bytes = sum(f.stat().st_size for f in common.iterdir())
            source = directory / 'staging.f32'
            canonical.tofile(source)
            expected = canonical.tobytes()
            variants = []
            for name, mapping, dedup in [('canonical-codec-only', None, False),
                    ('canonical-datrie-only', representatives, False), ('canonical-byte-dedup-only', None, True)]:
                artifact = directory / (name + '.men')
                started = time.perf_counter()
                storage = pack_native(source, artifact, dim * 4, block_rows=16, predictor='xor',
                                      representatives=mapping, deduplicate=dedup)
                with NativeRows(artifact, cfg.native_library) as native:
                    actual = native.rows(range(len(rows)))
                if actual != expected:
                    raise ValueError('isolated planner changed canonical row bytes')
                result = {'name': name, 'storage': storage, 'all_canonical_rows_byte_exact': len(rows),
                          'decoded_sha256': hashlib.sha256(actual).hexdigest(),
                          'artifact_sha256': file_sha256(artifact),
                          'build_and_verify_seconds': time.perf_counter() - started}
                write_json(directory / (name + '.json'), result)
                result['complete_bytes'] = common_bytes + artifact.stat().st_size + (directory / (name + '.json')).stat().st_size
                variants.append(result)
            source.unlink()
            base = variants[0]['complete_bytes']
            report['runs'].append({'order': order, 'seed': seed, 'partition': partition.summary(),
                'source_checkpoint_sha256': file_sha256(checkpoint_path), 'canonical_sha256': hashlib.sha256(expected).hexdigest(),
                'variants': variants, 'datrie_extra_saving_over_same_canonical_codec': 1 - variants[1]['complete_bytes'] / base,
                'byte_dedup_extra_saving_over_same_canonical_codec': 1 - variants[2]['complete_bytes'] / base})
            write_json(cfg.output_dir / 'report.json', report)
    report['status'] = 'completed'
    write_json(cfg.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
