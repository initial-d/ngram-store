"""Materialize counted common files and audit stronger fixed-vocabulary controls.

Post-run audit, not validation-selected evidence. Precomputed word aggregates
are a stronger serving baseline within the fixed known-word scope. They do not
reproduce per-n-gram SGD updates or answer arbitrary certified group queries.
"""
import argparse
from collections import Counter
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

from .benchmark_cotraining import vector_metrics, write_json
from .benchmark_heldout_compression import split_tokens, evaluation_set, compose, score, summarize, deltas
from .engram_row_adapter import file_sha256
from .native_row_store import NativeRows, pack_native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-dir', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--training-dir', type=Path, required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('empty CUDA_VISIBLE_DEVICES required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    split = split_tokens(cfg.corpus.read_text())
    counts = Counter(split['train'])
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'python': sys.version,
              'torch': torch.__version__, 'numpy': np.__version__, 'os': platform.platform(),
              'category': 'post-run materialization and strong known-word serving baseline audit',
              'limitations': ['Known-word precomputation does not retain arbitrary n-gram queries or dense SGD parameter trajectories.',
                              'SGNS scores on known centers/targets only; not semantics, perplexity or LLM memory.',
                              'Precomputed query timings are warm in-memory selection, not whole-model inference.',
                              'This is a post-run audit; no validation selection or confirmatory budget claim.',
                              'No GPU touched; monetary/energy cost unmeasured.'], 'runs': []}
    for study in ('cached-results', 'dynamics-legacy', 'dynamics-average'):
        primary = json.loads((cfg.task_dir / study / 'report.json').read_text())
        for entry in primary['runs']:
            order, seed = entry['order'], entry['seed']
            convention = 'cached' if study == 'cached-results' else study.removeprefix('dynamics-')
            directory = cfg.task_dir / study / f'{convention}-order-{order}-seed-{seed}'
            source = (cfg.task_dir / 'source/artifacts/heldout-compression-20261024' / f'order-{order}-seed-{seed}/reference.pt'
                      if study == 'cached-results' else directory / 'reference.pt')
            checkpoint = torch.load(source, map_location='cpu', weights_only=True)
            rows, outputs = checkpoint['input_rows'], checkpoint['output_rows']
            if hashlib.sha256(rows.numpy().tobytes()).hexdigest() != entry['source_input_sha256'] or \
                    hashlib.sha256(outputs.numpy().tobytes()).hexdigest() != entry['source_output_sha256']:
                raise ValueError('source checkpoint provenance mismatch')
            common = directory / 'common'
            common.mkdir(exist_ok=False)
            keys = json.dumps([checkpoint['words'], checkpoint['grams']], ensure_ascii=True).encode()
            (common / 'keys.json').write_bytes(keys)
            outputs.numpy().astype('<f4').tofile(common / 'output.f32')
            actual_common_bytes = sum(f.stat().st_size for f in common.iterdir())
            if actual_common_bytes != entry['common_output_and_keys_bytes']:
                raise ValueError('counted common bytes differ from materialized files')
            hidden = compose(rows, checkpoint['bags'], checkpoint['lengths'])
            baseline_dir = directory / 'precomputed-known-words'
            baseline_dir.mkdir()
            raw_path = baseline_dir / 'word-hidden.f32'
            hidden.numpy().astype('<f4').tofile(raw_path)
            packed = pack_native(raw_path, baseline_dir / 'word-hidden.men', hidden.shape[1] * 4,
                                 block_rows=max(1, 8192 // (hidden.shape[1] * 4)), deduplicate=False, predictor='xor')
            with NativeRows(baseline_dir / 'word-hidden.men', cfg.native_library) as native:
                if native.rows(range(len(hidden))) != hidden.numpy().tobytes():
                    raise ValueError('precomputed aggregate native bytes mismatch')
            # Raw and encoded tables are alternative layouts, never counted together.
            word_keys = json.dumps(checkpoint['words']).encode()
            (baseline_dir / 'words.json').write_bytes(word_keys)
            vocabulary_cost = len(word_keys)
            timings = []
            ids = np.random.default_rng(42).integers(len(hidden), size=1200)
            for _ in range(6):
                begin = time.perf_counter()
                selected = hidden[torch.from_numpy(ids)].numpy()
                timings.append(time.perf_counter() - begin)
            item = {'study': study, 'order': order, 'seed': seed, 'materialized_common_bytes': actual_common_bytes,
                    'source_checkpoint_sha256': file_sha256(source), 'precomputed': {
                        'raw_word_table_bytes': hidden.numel() * 4, 'encoded_word_table_bytes': packed['artifact_bytes'],
                        'complete_raw_word_output_key_bytes': hidden.numel() * 4 + outputs.numel() * 4 + vocabulary_cost,
                        'complete_encoded_word_output_key_bytes': packed['artifact_bytes'] + outputs.numel() * 4 + vocabulary_cost,
                        'aggregate_vectors_byte_exact_to_source_compose': True, 'warm_1200_word_selection_seconds': timings,
                        'scope': 'fixed known-word inference only; no original n-gram row lookup or matched SGD parameterization'}}
            if study != 'cached-results':
                tied = torch.load(directory / 'tied-reference.pt', map_location='cpu', weights_only=True)
                tied_hidden = compose(tied['input_rows'], checkpoint['bags'], checkpoint['lengths'])
                quality = {}
                for name in ('validation', 'test'):
                    evaluation = evaluation_set(split[name], checkpoint['words'], counts, 8192, 5, 42 + (name == 'test'))
                    original = score(hidden, outputs, evaluation)
                    candidate = score(tied_hidden, tied['output_rows'], evaluation)
                    quality[name] = {'original': summarize(original), 'tied': summarize(candidate),
                                     'delta': deltas(candidate, original), 'evaluation_sha256': evaluation['sha256']}
                item['matched_tied_training_quality'] = quality
                item['matched_tied_training_aggregate_fidelity'] = vector_metrics(hidden.numpy(), tied_hidden.numpy())
                initial = checkpoint['initial_rows'].numpy().view(np.uint32)
                final = rows.numpy().view(np.uint32)
                item['actual_changed_input_rows'] = int(np.any(initial != final, axis=1).sum())
                item['registered_input_rows'] = len(rows)
            report['runs'].append(item)
    large = json.loads((cfg.task_dir / 'large-results/report.json').read_text())
    common = cfg.task_dir / 'large-results/common'
    common.mkdir()
    for name in ('keys.json', 'trained-source-ids.u64'):
        shutil.copyfile(cfg.training_dir / name, common / name)
    common_bytes = sum(f.stat().st_size for f in common.iterdir())
    if common_bytes != large['common_keys_and_original_training_ids_bytes']:
        raise ValueError('large materialized common cost mismatch')
    report['large_materialized_common_bytes'] = common_bytes
    training = json.loads((cfg.training_dir / 'training.json').read_text())
    corpus_zip = Path(training['settings']['corpus_zip'])
    if file_sha256(corpus_zip) != training['zip_sha256']:
        raise ValueError('large corpus provenance mismatch')
    with zipfile.ZipFile(corpus_zip) as archive:
        raw = archive.read('enwik8')
    cutoff = int(len(raw) * .9)
    text = re.sub(r'<[^>]*>', ' ', raw[:cutoff].decode('utf-8', errors='replace')).lower()
    counts = Counter(w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text) if 2 <= len(w) <= 64)
    words = sorted(w for w, count in counts.items() if count >= training['settings']['min_count'])
    heldout = re.findall(r"[a-z]+(?:'[a-z]+)?", re.sub(r'<[^>]*>', ' ', raw[cutoff:].decode('utf-8', errors='replace')).lower())
    del raw, text
    keys = json.loads((common / 'keys.json').read_text())
    lookup = {key: i for i, key in enumerate(keys)}
    from .train_ngram_similarity import ngrams
    bags = [[lookup[g] for g in ngrams(word, training['settings']['order']) if g in lookup] for word in words]
    flat = np.asarray([i for bag in bags for i in bag], dtype=np.int64)
    offsets = np.r_[0, np.cumsum([len(b) for b in bags])].astype(np.int64)
    source_rows = np.memmap(cfg.training_dir / 'trained.f32', mode='r', dtype='<f4', shape=(len(keys), training['dim']))
    hidden = torch.nn.functional.embedding_bag(torch.from_numpy(flat), torch.from_numpy(source_rows),
                 torch.from_numpy(offsets), mode='mean', include_last_offset=True).numpy()
    baseline = cfg.task_dir / 'large-results/precomputed-known-words'
    baseline.mkdir()
    hidden.astype('<f4').tofile(baseline / 'word-hidden.f32')
    packed = pack_native(baseline / 'word-hidden.f32', baseline / 'word-hidden.men', training['dim'] * 4,
                         block_rows=max(1, 8192 // (training['dim'] * 4)), deduplicate=False, predictor='xor')
    word_keys = json.dumps(words).encode()
    (baseline / 'words.json').write_bytes(word_keys)
    available = sorted(set(heldout) & set(words))
    chosen = [available[i] for i in np.random.default_rng(42).choice(len(available), min(512, len(available)), replace=False)]
    word_ids = {word: i for i, word in enumerate(words)}
    selected = np.asarray([word_ids[word] for word in chosen], dtype=np.int64)
    original = np.asarray([source_rows[bags[i]].astype(np.float64).mean(0) if bags[i] else np.zeros(training['dim']) for i in selected], dtype=np.float32)
    timings = []
    for _ in range(9):
        begin = time.perf_counter()
        values = hidden[selected]
        timings.append(time.perf_counter() - begin)
    with NativeRows(baseline / 'word-hidden.men', cfg.native_library) as native:
        if native.rows(range(len(hidden))) != hidden.astype('<f4').tobytes():
            raise ValueError('large precomputed byte verification failed')
    report['large_precomputed'] = {'words': len(words), 'scope': 'known-word n-gram-subset mean only; word/output model unavailable',
        'complete_raw_word_table_and_key_bytes': hidden.nbytes + len(word_keys),
        'complete_encoded_word_table_and_key_bytes': packed['artifact_bytes'] + len(word_keys),
        'aggregate_fidelity': vector_metrics(original, values), 'warm_512_word_selection_seconds': timings,
        'all_precomputed_rows_byte_verified': len(hidden)}
    report['status'] = 'completed'
    write_json(cfg.task_dir / 'final-audit.json', report)


if __name__ == '__main__':
    main()
