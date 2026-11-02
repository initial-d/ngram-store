"""Paired CPU trace: retain certified row sharing but avoid marginal codec work.

Reuses verified trained canonical rows and exact query IDs. Raw blocks retain
the native format/forwarding API. This aggregate-domain transformation changes
original vectors; it is not original-row lossless or an LLM quality experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .benchmark_cotraining import vector_metrics, write_json
from .cotraining_groups import BagPartition, CertifiedBagRows
from .engram_row_adapter import file_sha256
from .native_row_store import NativeRows, pack_native
from .train_ngram_similarity import ngrams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-dir', type=Path, required=True)
    p.add_argument('--training-dir', type=Path, required=True)
    p.add_argument('--native-library', type=Path, required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = cfg.task_dir / 'large-results'
    old = json.loads((root / 'report.json').read_text())
    training = json.loads((cfg.training_dir / 'training.json').read_text())
    output = root / 'certified-raw-blocks'
    output.mkdir(exist_ok=False)
    partition = BagPartition.load(root / 'certified-groups/domain.npz')
    dim = training['dim']
    reps = partition.representatives()
    roots = np.unique(reps).tolist()
    # Group IDs were assigned in first-original-ID order by the exact fitter.
    if not np.array_equal(partition.mapping[roots], np.arange(len(roots))):
        raise ValueError('unexpected group order')
    means = np.empty((len(roots), dim), dtype='<f4')
    with NativeRows(root / 'certified-groups/table.men', cfg.native_library) as native:
        for begin in range(0, len(roots), 4096):
            ids = roots[begin:begin + 4096]
            means[begin:begin + len(ids)] = np.frombuffer(native.rows(ids), dtype='<f4').reshape(-1, dim)
    staging = output / 'staging.f32'
    with staging.open('wb') as stream:
        for begin in range(0, len(partition.mapping), 4096):
            means[partition.mapping[begin:begin + 4096]].tofile(stream)
    started = time.perf_counter()
    storage = pack_native(staging, output / 'table.men', dim * 4, block_rows=8, deduplicate=False,
                          representatives=reps.tolist(), predictor='xor', compress=False)
    decoded = hashlib.sha256()
    with NativeRows(output / 'table.men', cfg.native_library) as native:
        for begin in range(0, len(partition.mapping), 4096):
            ids = list(range(begin, min(begin + 4096, len(partition.mapping))))
            actual = native.rows(ids)
            if actual != means[partition.mapping[ids]].tobytes():
                raise ValueError('raw-block canonical byte mismatch')
            decoded.update(actual)
    source_manifest = json.loads((root / 'certified-groups/manifest.json').read_text())
    if decoded.hexdigest() != source_manifest['decoded_sha256']:
        raise ValueError('encoded/raw canonical digests differ')
    staging.unlink()
    shutil.copyfile(root / 'certified-groups/domain.npz', output / 'domain.npz')
    manifest = {**source_manifest, 'name': 'certified-raw-blocks', 'storage': storage,
                'artifact_sha256': file_sha256(output / 'table.men'),
                'build_and_verification_seconds': time.perf_counter() - started}
    write_json(output / 'manifest.json', manifest)
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
              'torch': torch.__version__, 'numpy': np.__version__, 'os': platform.platform(),
              'device': 'cpu', 'threads': 1, 'all_canonical_rows_byte_verified': len(partition.mapping),
              'decoded_sha256': decoded.hexdigest(), 'source_report_sha256': file_sha256(root / 'report.json'),
              'complete_subset_bytes': old['common_keys_and_original_training_ids_bytes'] + sum(f.stat().st_size for f in output.iterdir()),
              'limitations': ['Same finite SGNS subset and certified aggregate domain; source row bytes change.',
                              'Paired trace on a shared CPU with warm OS pages; preprocessing excluded from query timings.',
                              'Precomputed known-word mean is a restricted serving baseline, not matched training/arbitrary n-gram lookup.',
                              'No full model quality/LLM/GPU claim; monetary/energy cost unmeasured.']}
    words = json.loads((root / 'precomputed-known-words/words.json').read_text())
    keys = json.loads((root / 'common/keys.json').read_text())
    lookup = {key: i for i, key in enumerate(keys)}
    with zipfile.ZipFile(training['settings']['corpus_zip']) as archive:
        raw = archive.read('enwik8')
    cutoff = int(.9 * len(raw))
    heldout = re.findall(r"[a-z]+(?:'[a-z]+)?", re.sub(r'<[^>]*>', ' ', raw[cutoff:].decode('utf-8', errors='replace')).lower())
    available = sorted(set(heldout) & set(words))
    rng = np.random.default_rng(42)
    chosen = [available[i] for i in rng.choice(len(available), min(512, len(available)), replace=False)]
    bags = [[lookup[g] for g in ngrams(word, training['settings']['order']) if g in lookup] for word in chosen]
    raw_ids = np.asarray([i for bag in bags for i in bag], dtype='<u8')
    offsets = np.r_[0, np.cumsum([len(b) for b in bags])].astype('<u8')
    trace_hash = hashlib.sha256(raw_ids.tobytes() + offsets.tobytes()).hexdigest()
    if trace_hash != old['query_bags_sha256']:
        raise ValueError('query trace differs from primary study')
    report['query_bags_sha256'] = trace_hash
    source_rows = np.memmap(cfg.training_dir / 'trained.f32', mode='r', dtype='<f4', shape=(len(keys), dim))
    source_tensor = torch.from_numpy(source_rows)
    torch_ids, torch_offsets = torch.from_numpy(raw_ids.astype(np.int64)), torch.from_numpy(offsets.astype(np.int64))
    word_ids = {word: i for i, word in enumerate(words)}
    selection = np.asarray([word_ids[word] for word in chosen])
    precomputed = np.memmap(root / 'precomputed-known-words/word-hidden.f32', mode='r', dtype='<f4', shape=(len(words), dim))
    reference = np.asarray([source_rows[b].astype(np.float64).mean(0) for b in bags], dtype=np.float32)
    timings = {key: [] for key in ('raw-mmap', 'encoded-certified', 'raw-certified', 'precomputed-known-words')}
    with CertifiedBagRows(root / 'certified-groups', cfg.native_library) as encoded, \
            CertifiedBagRows(output, cfg.native_library) as direct:
        encoded.table.set_cache_bytes(4 << 20)
        direct.table.set_cache_bytes(4 << 20)
        plan = direct.compile(bags)
        report['plan'] = plan.summary()
        for _ in range(9):
            for name in rng.permutation(list(timings)):
                begin = time.perf_counter()
                if name == 'encoded-certified':
                    values = encoded.evaluate(plan, mean=True)
                elif name == 'raw-certified':
                    values = direct.evaluate(plan, mean=True)
                elif name == 'raw-mmap':
                    values = F.embedding_bag(torch_ids, source_tensor, torch_offsets, mode='mean', include_last_offset=True).numpy()
                else:
                    values = precomputed[selection]
                timings[name].append(time.perf_counter() - begin)
        report['aggregate_fidelity'] = vector_metrics(reference, direct.evaluate(plan, mean=True))
        report['raw_native_decompressions'] = direct.table._counter('last_decompressions')
    report['query_samples_seconds'] = timings
    report['complete_subset_and_plan_bytes'] = report['complete_subset_bytes'] + report['plan']['plan_bytes']
    report['status'] = 'completed'
    write_json(cfg.task_dir / 'raw-block-evidence.json', report)


if __name__ == '__main__':
    main()
