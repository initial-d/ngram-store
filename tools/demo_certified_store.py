"""Offline NumPy SGNS training → native compaction → certified queries.

The included repository-authored text is MIT. This small example establishes
integration, not representative capacity, quality or performance evidence.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

import numpy as np

from .certified_store import compact_native
from .cotraining_groups import BagPartition, CertifiedBagRows
from .external_cotraining import file_hash


def demo(output, *, native_library=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        corpus = Path(__file__).with_name('certified_sample.txt')
        text = corpus.read_text()
        shutil.copy(corpus, output / 'corpus.txt')
        tokens = re.findall('[a-z]+', text.lower())
        words = sorted(set(tokens))
        word_ids = {w: i for i, w in enumerate(words)}
        def grams(word):
            bounded = '<' + word + '>'
            return [bounded[i:i + n] for i in range(len(bounded)) for n in range(2, 7)
                    if i + n <= len(bounded)]
        keys = sorted({g for w in words for g in grams(w)})
        offset = len(words)
        gram_ids = {g: offset + i for i, g in enumerate(keys)}
        bags = [[word_ids[w], *[gram_ids[g] for g in grams(w)]] for w in words]
        sequence = [word_ids[w] for w in tokens]
        pairs = [(sequence[i], sequence[j]) for i in range(len(sequence))
                 for j in range(max(0, i - 2), min(len(sequence), i + 3)) if i != j]
        seed, dim, steps, rate = 7, 32, 1200, .025
        rng = np.random.default_rng(seed)
        rows = rng.uniform(-1 / dim, 1 / dim, (offset + len(keys), dim)).astype('<f4')
        initial = rows.copy()
        targets = np.zeros((len(words), dim), dtype='<f4')
        counts = Counter(tokens)
        probability = np.asarray([counts[w] ** .5 for w in words])
        probability /= probability.sum()
        losses = []
        for step in range(steps):
            center, target = pairs[int(rng.integers(len(pairs)))]
            bag = bags[center]
            hidden = rows[bag].sum(axis=0) / len(bag)
            negative = rng.choice(len(words), 5, p=probability)
            positive_row, negative_rows = targets[target].copy(), targets[negative].copy()
            positive = float(hidden @ positive_row)
            logits = negative_rows @ hidden
            dpositive = float(1 / (1 + np.exp(-np.clip(positive, -60, 60)))) - 1
            dnegative = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
            gradient = dpositive * positive_row + (dnegative[:, None] * negative_rows).sum(axis=0)
            # Historical average-forward/unscaled-row SGD convention.
            np.add.at(rows, bag, -rate * gradient)
            np.add.at(targets, target, -rate * dpositive * hidden)
            np.add.at(targets, negative, -rate * dnegative[:, None] * hidden)
            if step % 200 == 0 or step == steps - 1:
                losses.append({'step': step, 'sample_sgns_loss': float(np.logaddexp(0, -positive)
                               + np.logaddexp(0, logits).sum())})
        if not np.isfinite(rows).all() or np.array_equal(rows, initial):
            raise ValueError('example training failed')
        rows.tofile(output / 'vectors.f32')
        targets.tofile(output / 'output.f32')
        np.asarray([row for bag in bags for row in bag], dtype='<u8').tofile(output / 'ids.u64')
        np.asarray([0, *np.cumsum([len(b) for b in bags])], dtype='<u8').tofile(output / 'offsets.u64')
        np.arange(offset, dtype='<u8').tofile(output / 'protected.u64')
        (output / 'keys.json').write_text(json.dumps({'words': words, 'ngrams': keys}, indent=2) + '\n')
        queries = bags + [[], bags[0] * 2]
        (output / 'queries.json').write_text(json.dumps(queries) + '\n')
        build = compact_native(output / 'vectors.f32', dim, output / 'ids.u64', output / 'offsets.u64',
            output / 'memory', protected_ids=output / 'protected.u64', memory_bytes=8 << 20)
        partition = BagPartition.fit(bags, len(rows), protected=range(offset))
        expected = partition.means(rows)[partition.mapping]
        with CertifiedBagRows(output / 'memory', native_library) as store:
            if store.table.rows(list(range(len(rows)))) != expected.tobytes():
                raise ValueError('native rows differ from the legacy projection')
            actual = store.aggregate(queries, mean=True)
            original = np.asarray([rows[b].astype(np.float64).mean(axis=0) if b else np.zeros(dim)
                                   for b in queries], dtype=np.float32)
            errors = actual.astype(np.float64) - original
            if not np.allclose(actual, original, rtol=1e-5, atol=1e-7):
                raise ValueError('certified example aggregates differ beyond the example rounding tolerance')
            selected = next(i for i, group in enumerate(partition.mapping) if partition.sizes[group] > 1)
            try:
                store.aggregate([[selected]])
            except ValueError:
                pass
            else:
                raise ValueError('incomplete group was not rejected')
        (output / 'rejected-query.json').write_text(json.dumps([[selected]]) + '\n')
        (output / 'outputs.json').write_text(json.dumps(actual.tolist()) + '\n')
        report = {'status': 'passed', 'category': 'included-text actual CPU training and integration example',
            'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
            'seed': seed, 'dim': dim, 'steps': steps, 'lr': rate, 'gradient_rule': 'legacy unscaled row SGD',
            'torch_loaded': 'torch' in sys.modules, 'source_corpus_sha256': file_hash(corpus),
            'source_code_sha256': file_hash(Path(__file__)), 'rows': len(rows),
            'changed_rows': int(np.any(rows != initial, axis=1).sum()), 'training_samples': losses,
            'partition': partition.summary(), 'complete_serving_bytes': build['complete_serving_bytes'],
            'every_native_row_identical_to_legacy_projection': True, 'queries': len(queries),
            'maximum_aggregate_absolute_error_vs_source_fp64': float(np.max(np.abs(errors))),
            'incomplete_group_rejected': selected, 'seconds': time.perf_counter() - started,
            'limitations': ['Small MIT repository-authored text; no representative compression/quality/performance claim.',
                'Actual SGNS updates after independent initialization; no random-vector similarity premise.',
                'Projection changes rows; registered complete aggregates only; FP32 rounding measured.',
                'Example retains source files for reproduction; serving directory has 3 files.']}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    except BaseException:
        shutil.rmtree(output)
        raise
