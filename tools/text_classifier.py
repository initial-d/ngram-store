"""Frozen text -> learned n-gram bags -> linear classifier, NumPy/C serving.

Prediction preservation certifies the declared FP32 source model in real
arithmetic. It is separate from classification accuracy and source-row losslessness.
"""
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from .budgeted_bags import BudgetedBagRows, _gamma, _norm_upper, _up, _down, _sum_upper
from .cotraining_groups import _count_bags
from .external_cotraining import file_hash


def ngrams(word, maximum):
    bounded = '<' + word + '>'
    return [bounded[i:i+n] for i in range(len(bounded))
            for n in range(2, maximum+1) if i+n <= len(bounded)]


def tokenize(text, maximum=128):
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return words[:maximum], len(words) > maximum


def reduce_words(reader, bags, dim):
    """All direct codec controls share unique-ID gathering and FP64 reduction."""
    counted = _count_bags(bags)
    ids, counts, bids = (counted[k] for k in ('rows', 'counts', 'bags'))
    unique, inverse = np.unique(ids, return_inverse=True)
    rows = reader.rows(unique)
    values = rows[inverse].astype(np.float64) * counts[:, None]
    output = np.zeros((len(bags), dim), dtype='<f4')
    if len(ids):
        starts = np.r_[0, np.flatnonzero(bids[1:] != bids[:-1]) + 1]
        output[bids[starts]] = np.add.reduceat(values, starts, axis=0)
    output /= np.maximum(counted['lengths'], 1)[:, None]
    if not np.isfinite(output).all():
        raise ValueError('nonfinite word features')
    return output


def linear_preservation(vectors, bounds, head, bias):
    """Conservative interval proof for argmax of an FP32-source linear model.

vectors are computed binary64 document features; bounds include their deviation
from exact source features. Enclose both evaluated logits and pairwise margins.
"""
    vectors, bounds = np.asarray(vectors, '<f8'), np.asarray(bounds, '<f8')
    head, bias = np.asarray(head, '<f8'), np.asarray(bias, '<f8')
    if vectors.ndim != 2 or head.ndim != 2 or head.shape[1] != vectors.shape[1] or bias.shape != (len(head),):
        raise ValueError('matching feature/head/bias shapes required')
    if bounds.shape != (len(vectors),) or np.any(bounds < 0) or np.isnan(bounds).any():
        raise ValueError('nonnegative document bounds required')
    if not all(np.isfinite(a).all() for a in (vectors, head, bias)):
        raise ValueError('finite linear classifier required')
    if any(not np.array_equal(a, a.astype('<f4').astype('<f8')) for a in (head, bias)):
        raise ValueError('head and bias must be exactly promoted FP32 source values')
    logits = vectors @ head.T + bias
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite logits')
    # A d-term dot followed by bias: gamma(d+2) also covers rounded products.
    magnitude = np.broadcast_to(_up(np.abs(bias)), logits.shape).copy()
    for j in range(vectors.shape[1]):
        magnitude = _up(magnitude + _up(np.abs(vectors[:, j, None]) * np.abs(head[None, :, j])))
    rounding = _up(float(_gamma(vectors.shape[1] + 2)) * magnitude
                   + _up((vectors.shape[1] + 2) * 2.**-1074))
    winner = logits.argmax(1)
    preserved = np.ones(len(vectors), dtype=bool)
    for k in range(len(head)):
        selected = winner == k
        for j in range(len(head)):
            if k == j:
                continue
            norm = float(_norm_upper(_up(np.abs(head[k] - head[j]))))
            uncertainty = np.zeros(len(vectors)) if norm == 0 else _up(norm * bounds)
            uncertainty = _up(uncertainty + _up(rounding[:, k] + rounding[:, j]))
            margin = _down(logits[:, k] - logits[:, j])
            preserved[selected] &= margin[selected] > uncertainty[selected]
    return logits, winner, preserved


class _RawRows:
    def __init__(self, path, shape, resident=False):
        self.array = (np.fromfile(path, dtype='<f4').reshape(shape) if resident else
                      np.memmap(path, mode='r', dtype='<f4', shape=shape))

    def rows(self, ids):
        return self.array[ids]

    def close(self):
        if isinstance(self.array, np.memmap):
            self.array._mmap.close()


class TextClassifier:
    """Hash-bound, immutable, single-worker classifier; no Torch or source fallback.

All nonempty texts receive predictions. `require_preserved=True` additionally
refuses a batch containing any uncertified prediction. This flag establishes
source-model agreement, not truth, confidence calibration or downstream safety.
"""
    def __init__(self, directory, *, native_library=None, cache_bytes=4 << 20):
        directory = Path(directory)
        self.reader = None
        self.maps = []
        metadata = json.loads((directory / 'classifier.json').read_text())
        if metadata.get('schema_version') != 1 or metadata.get('kind') != 'ngram-linear-text':
            raise ValueError('unsupported text classifier')
        for name, digest in metadata['files_sha256'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or file_hash(directory / path) != digest:
                raise ValueError('classifier file hash mismatch: ' + name)
        self.metadata = metadata
        self.dim = metadata['dim']
        self.backend = metadata['backend']
        self.classes = metadata['classes']
        head = np.load(directory / 'head.npz', allow_pickle=False)
        self.head, self.bias = head['weight'], head['bias']
        head.close()
        lexicon = json.loads((directory / 'lexicon.json').read_text())
        self.maximum = lexicon['max_tokens']
        self.order = lexicon['max_order']
        self.words = {word: i for i, word in enumerate(lexicon['tokens'])}
        self.grams = {gram: lexicon['whole_word_rows'] + i for i, gram in enumerate(lexicon['grams'])}
        try:
            if self.backend == 'precomputed-words':
                self.reader = _RawRows(directory / 'word-vectors.f32', (len(self.words), self.dim))
                self.ids = self.offsets = None
            else:
                self.ids = np.memmap(directory / 'word-ids.u64', mode='r', dtype='<u8')
                self.offsets = np.memmap(directory / 'word-offsets.u64', mode='r', dtype='<u8')
                self.maps.extend((self.ids, self.offsets))
                if len(self.offsets) != len(self.words) + 1 or self.offsets[0] != 0 or self.offsets[-1] != len(self.ids):
                    raise ValueError('word bag offsets mismatch')
                if self.backend == 'ngram-bounded':
                    self.reader = BudgetedBagRows(directory / 'memory', directory / 'bounds', native_library)
                    self.reader.native.table.set_cache_bytes(cache_bytes)
                elif self.backend in ('source-resident', 'source-mmap'):
                    self.reader = _RawRows(directory / 'source.f32', (metadata['rows'], self.dim), self.backend == 'source-resident')
                elif self.backend in ('fp16', 'int8', 'ngram-int8'):
                    from .quantized_memory import QuantizedRows
                    reader = QuantizedRows(directory / 'memory', native_library, cache_bytes=cache_bytes)
                    reader.rows = reader.gather
                    self.reader = reader
                elif self.backend.startswith('faiss-'):
                    from .faiss_row_codec import FaissRows
                    self.reader = FaissRows(directory / 'memory')
                else:
                    raise ValueError('unknown text backend')
        except BaseException:
            self.close()
            raise

    def word_bag(self, word):
        index = self.words.get(word)
        if index is not None:
            return self.ids[int(self.offsets[index]):int(self.offsets[index + 1])]
        return [self.grams[g] for g in ngrams(word, self.order) if g in self.grams]

    def predict(self, texts, *, require_preserved=False):
        texts = list(texts)
        parsed = [tokenize(text, self.maximum) for text in texts]
        if any(not words for words, _ in parsed):
            raise ValueError('each text needs at least one supported token')
        unique = list(dict.fromkeys(word for words, _ in parsed for word in words))
        lookup = {word: i for i, word in enumerate(unique)}
        if self.backend == 'precomputed-words':
            features = np.zeros((len(unique), self.dim), dtype='<f4')
            selected = [(i, self.words[w]) for i, w in enumerate(unique) if w in self.words]
            if selected:
                positions, ids = zip(*selected)
                features[list(positions)] = self.reader.rows(np.asarray(ids))
            word_bounds = None
        else:
            bags = [self.word_bag(word) for word in unique]
            if self.backend == 'ngram-bounded':
                assessment = self.reader.assess(bags, mean=True)
                features, word_bounds = assessment.vectors, assessment.absolute_l2_bounds
            else:
                features = reduce_words(self.reader, bags, self.dim)
                word_bounds = None
        documents = np.zeros((len(texts), self.dim), dtype='<f8')
        bounds = np.full(len(texts), np.inf)
        for i, (words, _) in enumerate(parsed):
            counts = Counter(lookup[w] for w in words)
            positions = np.asarray(list(counts), dtype=np.int64)
            weights = np.asarray(list(counts.values()), dtype=np.float64)
            documents[i] = (features[positions].astype(np.float64) * weights[:, None]).sum(0) / len(words)
            if word_bounds is not None:
                residual = _up(_sum_upper(_up(weights * word_bounds[positions])) / len(words))
                magnitude = _sum_upper(_up(weights * _norm_upper(features[positions])))
                rounding = _up(_up(float(_gamma(len(positions) + 3)) * magnitude) / len(words))
                bounds[i] = _up(residual + rounding)
        logits, predicted, preserved = linear_preservation(documents, bounds, self.head, self.bias)
        if word_bounds is None:
            preserved[:] = False
        if require_preserved and not preserved.all():
            raise ValueError('one or more predictions lack a source-model preservation certificate')
        probabilities = np.exp(logits - logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        return {'predictions': predicted, 'probabilities': probabilities, 'logits': logits,
                'source_prediction_preserved': preserved, 'document_l2_bounds': bounds,
                'token_counts': np.asarray([len(w) for w, _ in parsed]),
                'oov_counts': np.asarray([sum(w not in self.words for w in words) for words, _ in parsed]),
                'truncated': np.asarray([t for _, t in parsed])}

    def close(self):
        if self.reader is not None:
            close = getattr(self.reader, 'close', None)
            if close:
                close()
            elif hasattr(self.reader, 'codes'):
                self.reader.codes._mmap.close()
            self.reader = None
        for array in self.maps:
            array._mmap.close()
        self.maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
