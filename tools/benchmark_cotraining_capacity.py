"""Actual large trained subset: incidence compaction, complete bytes and CPU bags.

Uses existing finite CPU SGNS data, never retrains or touches CUDA. This is a
model-side aggregate transformation, NOT a row-byte lossless checkpoint packer.
Unknown query compositions must be rejected. The full SGNS word/output model
was not exported, so this study cannot establish full-model quality/capacity.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .benchmark_cotraining import distribution, vector_metrics, write_json
from .cotraining_groups import BagPartition, CertifiedBagRows
from .engram_row_adapter import file_sha256
from .native_row_store import NativeRows, pack_native
from .train_ngram_similarity import ngrams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--corpus-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--query-words', type=int, default=512)
    parser.add_argument('--repeat', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    cfg = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or min(cfg.query_words, cfg.repeat) < 1:
        parser.error('CPU-only: empty CUDA_VISIBLE_DEVICES and positive query sizes required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    training = json.loads((cfg.training_dir / 'training.json').read_text())
    report = {'category': 'actual finite SGNS trained subset; certified bag aggregate transformation',
              'source_commit': os.environ.get('MEFT_SOURCE_COMMIT') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
              'numpy': np.__version__, 'torch': torch.__version__, 'os': platform.platform(), 'cpu': platform.processor(),
              'device': 'cpu', 'threads': 1, 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
              'limitations': ['One finite rare-center-coverage SGNS checkpoint, not convergence or LLM memory.',
                              'Registered changed-row subset, with original training-ID provenance; not a full-model export.',
                              'Individual source row bytes change; certified bag sums agree in real arithmetic only.',
                              'Queries use train-vocabulary known words occurring in held-out text; OOVs rejected and audited separately.',
                              'Complete bytes cover this subset, keys, source IDs, certificates, manifests and compiled plans; word/output model unavailable.',
                              'Shared CPU, OS page cache uncontrolled, not exclusive serving latency; no GPU process touched.',
                              'Wikipedia-derived enwik8; publish hashes/aggregates only; monetary/energy cost unmeasured.']}
    write_json(cfg.output_dir / 'protocol.json', report)
    for name, field in [('trained.f32', 'trained_sha256'), ('keys.json', 'keys_sha256'),
                        ('trained-source-ids.u64', 'source_row_ids_sha256')]:
        if file_sha256(cfg.training_dir / name) != training[field]:
            raise ValueError('source provenance mismatch: ' + name)
    if file_sha256(cfg.corpus_zip) != training['zip_sha256']:
        raise ValueError('corpus ZIP hash mismatch')
    with zipfile.ZipFile(cfg.corpus_zip) as archive:
        raw = archive.read('enwik8')
    if hashlib.sha256(raw).hexdigest() != training['corpus_raw_sha256']:
        raise ValueError('raw corpus hash mismatch')
    cutoff = int(len(raw) * .9)
    text = re.sub(r'<[^>]*>', ' ', raw[:cutoff].decode('utf-8', errors='replace'))
    counts = Counter(w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()) if 2 <= len(w) <= 64)
    words = sorted(w for w, count in counts.items() if count >= training['settings']['min_count'])
    if len(words) != training['vocabulary']:
        raise ValueError('training source vocabulary mismatch')
    heldout = re.sub(r'<[^>]*>', ' ', raw[cutoff:].decode('utf-8', errors='replace')).lower()
    heldout_words = re.findall(r"[a-z]+(?:'[a-z]+)?", heldout)
    del raw, text, heldout
    keys = json.loads((cfg.training_dir / 'keys.json').read_text())
    lookup = {key: i for i, key in enumerate(keys)}
    shape = (len(keys), training['dim'])
    rows = np.memmap(cfg.training_dir / 'trained.f32', mode='r', dtype='<f4', shape=shape)
    source_ids = np.fromfile(cfg.training_dir / 'trained-source-ids.u64', dtype='<u8')
    if len(source_ids) != len(rows) or len(np.unique(source_ids)) != len(rows):
        raise ValueError('invalid original training IDs')
    def word_bag(word):
        return [lookup[g] for g in ngrams(word, training['settings']['order']) if g in lookup]
    print('fitting exact bag incidence partition', flush=True)
    partition = BagPartition.fit((word_bag(word) for word in words), len(rows))
    report['partition'] = partition.summary()
    report['source'] = {k: training[k] for k in ('trained_sha256', 'keys_sha256', 'source_row_ids_sha256',
                       'ngram_rows', 'registered_ngram_rows', 'excluded_untrained_rows', 'dim', 'sampled_training_pairs')}
    report['partition_seconds'] = time.perf_counter() - started
    write_json(cfg.output_dir / 'progress.json', report)
    print(json.dumps(report['partition']), flush=True)
    means = partition.means(rows)
    report['means_seconds'] = time.perf_counter() - started - report['partition_seconds']
    canonical_path = cfg.output_dir / 'canonical-staging.f32'
    with canonical_path.open('wb') as stream:
        for begin in range(0, len(rows), 4096):
            means[partition.mapping[begin:begin + 4096]].astype('<f4').tofile(stream)
    original_norm2, error2, max_row_error = 0., 0., 0.
    for begin in range(0, len(rows), 4096):
        part = rows[begin:begin + 4096].astype(np.float64)
        delta = means[partition.mapping[begin:begin + len(part)]].astype(np.float64) - part
        norm2 = (part * part).sum(1)
        d2 = (delta * delta).sum(1)
        original_norm2 += float(norm2.sum())
        error2 += float(d2.sum())
        max_row_error = max(max_row_error, float(np.sqrt(d2 / np.maximum(norm2, 1e-30)).max()))
    report['source_row_error'] = {'global_relative_l2': (error2 / original_norm2) ** .5,
                                  'maximum_row_relative_l2': max_row_error}
    canonical_digest = file_sha256(canonical_path)
    common_bytes = (cfg.training_dir / 'keys.json').stat().st_size + (cfg.training_dir / 'trained-source-ids.u64').stat().st_size
    report['common_keys_and_original_training_ids_bytes'] = common_bytes
    variants = []
    for name, source, representatives, domain in [('identity', cfg.training_dir / 'trained.f32', None, None),
                ('certified-groups', canonical_path, partition.representatives().tolist(), partition)]:
        path = cfg.output_dir / name
        path.mkdir()
        build_started = time.perf_counter()
        packed = pack_native(source, path / 'table.men', training['dim'] * 4, block_rows=8,
                             deduplicate=False, representatives=representatives, predictor='xor')
        digest = hashlib.sha256()
        with NativeRows(path / 'table.men', cfg.native_library) as native:
            for begin in range(0, len(rows), 4096):
                stop = min(len(rows), begin + 4096)
                actual = native.rows(range(begin, stop))
                expected = (rows[begin:stop] if domain is None else means[partition.mapping[begin:stop]]).astype('<f4').tobytes()
                if actual != expected:
                    raise AssertionError('native representative byte mismatch')
                digest.update(actual)
        if digest.hexdigest() != (training['trained_sha256'] if domain is None else canonical_digest):
            raise AssertionError('decoded table digest mismatch')
        manifest = {'name': name, 'source_row_byte_preserving': domain is None,
                    'native_decodes_declared_rows_byte_exactly': True, 'all_original_ids_checked': len(rows),
                    'decoded_sha256': digest.hexdigest(), 'artifact_sha256': file_sha256(path / 'table.men'),
                    'storage': packed, 'build_and_verification_seconds': time.perf_counter() - build_started}
        if domain is not None:
            partition.save(path / 'domain.npz')
            manifest['partition'] = partition.summary()
            manifest['contract'] = 'certified bag aggregates in real arithmetic; FP rounding measured; reject uncertified bags'
        write_json(path / 'manifest.json', manifest)
        manifest['complete_subset_bytes'] = common_bytes + sum(f.stat().st_size for f in path.iterdir() if f.is_file())
        variants.append(manifest)
        print(json.dumps({'variant': name, 'bytes': manifest['complete_subset_bytes']}), flush=True)
    report['variants'] = variants
    canonical_path.unlink()
    rng = np.random.default_rng(cfg.seed)
    vocabulary = set(words)
    available = sorted(set(heldout_words) & vocabulary)
    chosen = [available[i] for i in rng.choice(len(available), min(cfg.query_words, len(available)), replace=False)]
    bags = [word_bag(word) for word in chosen]
    raw_ids = np.asarray([i for bag in bags for i in bag], dtype='<u8')
    offsets = np.r_[0, np.cumsum([len(b) for b in bags])].astype('<u8')
    raw_reference = np.asarray([rows[bag].astype(np.float64).mean(0) if bag else np.zeros(shape[1]) for bag in bags], dtype=np.float32)
    raw_tensor = torch.from_numpy(rows)
    torch_ids, torch_offsets = torch.from_numpy(raw_ids.astype(np.int64)), torch.from_numpy(offsets.astype(np.int64))
    dense_times, native_times, grouped_times = [], [], []
    original = np.empty_like(raw_reference)
    with NativeRows(cfg.output_dir / 'identity/table.men', cfg.native_library, cache_bytes=4 << 20) as native, \
            CertifiedBagRows(cfg.output_dir / 'certified-groups', cfg.native_library) as grouped:
        grouped.table.set_cache_bytes(4 << 20)
        compile_started = time.perf_counter()
        plan = grouped.compile(bags)
        report['query_plan'] = {**plan.summary(), 'compile_seconds': time.perf_counter() - compile_started}
        for repeat in range(cfg.repeat + 1):
            for name in rng.permutation(['raw-mmap', 'identity-native', 'certified-native']):
                begin = time.perf_counter()
                if name == 'raw-mmap':
                    result = F.embedding_bag(torch_ids, raw_tensor, torch_offsets, mode='mean', include_last_offset=True).numpy()
                    dense_times.append(time.perf_counter() - begin)
                elif name == 'identity-native':
                    native.embedding_bag_into(raw_ids.ctypes.data, len(raw_ids), offsets.ctypes.data, len(bags),
                        1, True, (1 << 64) - 1, 64, original.ctypes.data, original.size)
                    native_times.append(time.perf_counter() - begin)
                else:
                    result = grouped.evaluate(plan, mean=True)
                    grouped_times.append(time.perf_counter() - begin)
        report['compiled_query_fidelity'] = vector_metrics(raw_reference, grouped.evaluate(plan, mean=True))
        report['identity_query_fidelity'] = vector_metrics(raw_reference, original)
    report['query_samples_seconds'] = {'raw-mmap': dense_times, 'identity-native': native_times, 'certified-native': grouped_times}
    report['query_bags_sha256'] = hashlib.sha256(raw_ids.tobytes() + offsets.tobytes()).hexdigest()
    oov = sorted(set(heldout_words) - vocabulary)
    chosen_oov = [oov[i] for i in rng.choice(len(oov), min(4096, len(oov)), replace=False)]
    safe, errors = 0, []
    for word in chosen_oov:
        bag = word_bag(word)
        safe += int(partition.certifies(bag))
        if bag:
            old, new = rows[bag].astype(np.float64).mean(0), means[partition.mapping[bag]].astype(np.float64).mean(0)
            errors.append(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-30))
    report['oov'] = {'heldout_oov_types': len(oov), 'audited_types': len(chosen_oov), 'certified': safe,
                     'rejected': len(chosen_oov) - safe, 'unsafe_unconditional_projection_relative_l2': distribution(errors)}
    report['raw_subset_and_common_bytes'] = rows.nbytes + common_bytes
    report['complete_certified_subset_and_plan_bytes'] = variants[1]['complete_subset_bytes'] + report['query_plan']['plan_bytes']
    report['wall_seconds'] = time.perf_counter() - started
    report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report['status'] = 'completed'
    write_json(cfg.output_dir / 'report.json', report)
    print(json.dumps({'status': 'completed', 'seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
