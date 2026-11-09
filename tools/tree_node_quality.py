"""Complete CPU node-sharing/codec frontier with OOV inputs and locked audits.

Model weights and lexical inputs stay in the private run directory. Publication
uses numeric aggregates, source hashes and paired token-block sufficient stats.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

GATES = (.99, .98, .97, .95, .90)
SPLITS = ('validation', 'test', 'confirmation')
STRATA = ('all', 'known', 'oov_nonempty', 'oov_empty')
BUDGETS = (.0001, .001, .005, .01, .02)
ROOT = Path(__file__).resolve().parents[1]


def load(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    temporary = path.with_suffix(path.suffix + '.tmp')
    if path.suffix == '.gz':
        temporary.write_bytes(gzip.compress(data, mtime=0))
    else:
        temporary.write_bytes(data)
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def memory():
    values = {'peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
    if sys.platform == 'linux':
        stats = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        for label, key in (('rss_bytes', 'VmRSS'), ('anonymous_bytes', 'RssAnon'), ('file_backed_bytes', 'RssFile')):
            values[label] = int(stats[key].split()[0]) * 1024
    return values


def environment():
    import faiss
    from importlib.metadata import version
    return {'commit': os.environ.get('MEFT_SOURCE_COMMIT') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': digest(__file__), 'command': shlex.join([sys.executable, *sys.argv]),
        'python': sys.version, 'numpy': np.__version__, 'faiss': faiss.__version__, 'torch': version('torch'),
        'faiss_compile_options': faiss.get_compile_options(), 'os': platform.platform(),
        'cpu': platform.processor(), 'device': 'cpu', 'gpu': 'unused',
        'loadavg': list(os.getloadavg()), 'threads': 1}


def context_evaluation(tokens, words, lexical_ids, counts, seed):
    """All input centers; only registered output targets, without closing OOV gaps."""
    lookup = {w: i for i, w in enumerate(words)}
    triples = [(lexical_ids[word], lookup[tokens[j]], i)
        for i, word in enumerate(tokens)
        for j in range(max(0, i - 2), min(len(tokens), i + 3))
        if i != j and tokens[j] in lookup]
    if not triples:
        raise ValueError('segment has no registered context targets')
    triples = np.asarray(triples, dtype='<i8')
    probability = np.asarray([counts[w] ** .5 for w in words], dtype=np.float64)
    probability /= probability.sum()
    negative = np.random.default_rng(seed).choice(len(words), (len(triples), 5), p=probability)
    arrays = {'pairs': triples[:, :2], 'positions': triples[:, 2], 'negative': negative}
    metadata = {'pairs': len(triples), 'all_valid_context_pairs_scored': True,
        'tokens': len(tokens), 'registered_target_token_fraction': sum(w in lookup for w in tokens) / len(tokens),
        'negative_target_collision_fraction': float((negative == triples[:, 1:2]).mean()),
        'sha256': hashlib.sha256(b''.join(np.ascontiguousarray(arrays[k], dtype='<i8').tobytes()
            for k in ('pairs', 'positions', 'negative'))).hexdigest(), 'negative_seed': seed}
    return arrays, metadata


def strata_masks(evaluation, word_ids, nonempty):
    ids = evaluation['pairs'][:, 0]
    known = word_ids[ids] >= 0
    return {'all': np.ones(len(ids), dtype=bool), 'known': known,
            'oov_nonempty': ~known & nonempty[ids], 'oov_empty': ~known & ~nonempty[ids]}


def paired_blocks(candidate, baseline, positions, mask):
    group, inverse = np.unique(positions[mask] // 128, return_inverse=True)
    result = {'block_ids': group.tolist(), 'count': np.bincount(inverse).tolist()}
    for prefix, scores in (('candidate', candidate), ('baseline', baseline)):
        for metric in ('loss', 'discrimination'):
            result[prefix + '_' + metric] = np.bincount(inverse, weights=scores[metric][mask]).tolist()
    return result


def block_summary(blocks):
    n = sum(blocks['count'])
    if not n:
        return {'pairs': 0, 'loss_relative_change': None, 'discrimination_change': None}
    a, b = sum(blocks['candidate_loss']), sum(blocks['baseline_loss'])
    x, y = sum(blocks['candidate_discrimination']), sum(blocks['baseline_discrimination'])
    return {'pairs': n, 'sgns_loss_per_pair': a / n, 'baseline_sgns_loss_per_pair': b / n,
            'context_discrimination': x / n, 'baseline_context_discrimination': y / n,
            'loss_relative_change': a / b - 1, 'discrimination_change': (x - y) / n}


def bootstrap(blocks, repeats=2000, seed=42):
    n = len(blocks['count'])
    if not n:
        return {'observed_blocks': 0, 'repeats': repeats, 'seed': seed, 'interval': None}
    rng = np.random.default_rng(seed)
    totals = {k: np.asarray(blocks[k], dtype=np.float64) for k in
        ('count', 'candidate_loss', 'baseline_loss', 'candidate_discrimination', 'baseline_discrimination')}
    samples = []
    for start in range(0, repeats, 100):
        ids = rng.integers(0, n, (min(100, repeats - start), n))
        loss = totals['candidate_loss'][ids].sum(1) / totals['baseline_loss'][ids].sum(1) - 1
        discrimination = (totals['candidate_discrimination'][ids].sum(1) - totals['baseline_discrimination'][ids].sum(1)) / totals['count'][ids].sum(1)
        samples.extend(np.column_stack((loss, discrimination)).tolist())
    return {'observed_blocks': n, 'block_tokens': 128, 'repeats': repeats, 'seed': seed,
        'loss_relative_change_95_percent_interval': np.quantile(np.asarray(samples)[:, 0], [.025, .975]).tolist(),
        'discrimination_change_95_percent_interval': np.quantile(np.asarray(samples)[:, 1], [.025, .975]).tolist(),
        'samples': samples, 'limitation': 'Pointwise within-text block approximation; no independent-corpus or simultaneous guarantee'}


def quality_gate(quality, budget):
    for stratum in ('all', 'oov_nonempty'):
        value = quality[stratum]
        if not value['pairs'] or value['loss_relative_change'] > budget or value['discrimination_change'] < -.005:
            return False
    return True


def select(candidates, budget, cached):
    eligible = [c for c in candidates if quality_gate(c['validation'][str(int(cached))], budget)]
    return min(eligible, key=lambda c: (c['model_bytes'], c['name'])) if eligible else None


class Reader:
    def __init__(self, job, *, cache_bytes=0):
        from .faiss_row_codec import FaissRows
        from .quantized_memory import QuantizedRows
        self.mapping = None
        self.reader = (FaissRows(job['directory']) if job['backend'] in ('faiss', 'dedup') else
                       QuantizedRows(job['directory'], job['native_library'], cache_bytes=cache_bytes))
        if job['backend'] == 'dedup':
            self.mapping = np.load(Path(job['directory']) / 'original-to-code.npy', mmap_mode='r')
        self.metadata = {'rows': self.reader.metadata.get('rows', self.reader.metadata.get('shape', [0])[0]),
                         'dim': self.reader.metadata.get('dim', self.reader.metadata.get('shape', [0, 0])[1])}
        if job['backend'] == 'dense':
            self.reader.table = np.array(self.reader.table)
        self.constant = job['backend'] == 'constant'
        if self.constant:
            self.metadata['rows'] = load(Path(job['common']) / 'metadata.json')['shape'][0]

    def rows(self, ids):
        from .quantized_memory import QuantizedRows
        if self.constant:
            return np.repeat(self.reader.gather(np.array([0], dtype=np.int64)), len(ids), axis=0)
        return self.reader.gather(ids) if isinstance(self.reader, QuantizedRows) else self.reader.rows(ids)

    def close(self):
        from .row_codec_utils import close_reader
        close_reader(self.reader)


def lookup_strings(strings, index, word_lookup, order, cached=False, index_kind='datrie'):
    words = [word_lookup.get(w, -1) for w in strings]
    grams = [[] if cached and i >= 0 else string_ngrams(w, order) for w, i in zip(strings, words)]
    lengths = np.cumsum([0] + list(map(len, grams)))
    flat = [g for group in grams for g in group]
    found = index.lookup(flat, index=index_kind)
    bags = [[r for r in found[lo:hi] if r is not None] for lo, hi in zip(lengths[:-1], lengths[1:])]
    return bags, words


def string_ngrams(word, order):
    """Same boundary/multiplicity contract as training, without importing Torch."""
    bounded = '<' + word + '>'
    return [bounded[i:i + n] for i in range(len(bounded))
            for n in range(2, order + 1) if i + n <= len(bounded)]


def plan_for(bags, word_ids, reader, cached=False):
    from .benchmark_faiss_cotraining import schedule
    used = [[] if cached and w >= 0 else b for b, w in zip(bags, word_ids)]
    return schedule(used, word_ids, code_mapping=reader.mapping)


def evaluate_hidden(reader, common, cached):
    from .benchmark_faiss_cotraining import evaluate
    bags = load(common / 'lexical-bags.json')
    words = np.load(common / 'lexical-word-ids.npy')
    values = np.load(common / ('cached-word-rows.npy' if cached else 'word-rows.npy'), mmap_mode='r')
    return evaluate(reader, plan_for(bags, words, reader, cached), values, batch_bags=64)


def score_candidate(hidden, common, split):
    from .benchmark_faiss_cotraining import objective
    evaluation = dict(np.load(common / (split + '-evaluation.npz')))
    baseline = dict(np.load(common / (split + '-source-scores.npz')))
    scores = objective(hidden, np.load(common / 'output-rows.npy', mmap_mode='r'), evaluation)
    masks = strata_masks(evaluation, np.load(common / 'lexical-word-ids.npy'), np.load(common / 'lexical-nonempty.npy'))
    blocks = {name: paired_blocks(scores, baseline, evaluation['positions'], mask) for name, mask in masks.items()}
    return {name: block_summary(b) for name, b in blocks.items()}, blocks


def serving_files(directory):
    return {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


def build_job(job_path):
    import faiss
    faiss.omp_set_num_threads(4)
    from .faiss_row_codec import FaissRows, fit_codec
    from .quantized_memory import encode_rows, decode_rows, pack_quantized
    from .structural_native import StructuralPlan
    job = load(job_path); common = Path(job['common']); directory = Path(job['directory'])
    metadata = load(common / 'metadata.json')
    source = np.load(common / 'source-rows.npy', mmap_mode='r')
    if array_digest(source) != metadata['source_row_sha256']:
        raise ValueError('source trained rows changed')
    started = time.perf_counter(); before = memory()
    mapping = np.arange(len(source), dtype=np.int64); order = None
    planning = None
    if job['planner'] in ('datrie', 'reference', 'layout'):
        keys = load(common / 'gram-keys.json')
        start = time.perf_counter()
        if job['planner'] == 'reference':
            from .benchmark_cosine_merge import structural_merge
            mapping = structural_merge(source, keys, job['cosine'], job['norm_ratio'])
            order = [i for i in sorted(range(len(keys)), key=lambda i: keys[i]) if mapping[i] == i] + [i for i in range(len(keys)) if mapping[i] != i]
        else:
            with StructuralPlan(source.tobytes(), source.shape[1], dict(enumerate(keys)), job['structural_library'],
                mode='layout' if job['planner'] == 'layout' else 'joint',
                cosine=job.get('cosine') or .99, norm_ratio=job.get('norm_ratio') or .99) as plan:
                mapping = np.asarray(plan.representatives, dtype=np.int64)
                order = plan.row_order
        if not np.array_equal(mapping[mapping], mapping):
            raise AssertionError('non-idempotent final representative map')
        if job['planner'] != 'layout':
            from .benchmark_cosine_merge import fidelity
            fidelity_before_quantization = fidelity(source, source[mapping])
            if (fidelity_before_quantization['minimum_cosine'] < job['cosine'] - 1e-7 or
                fidelity_before_quantization['max_relative_norm_change'] > 1 / job['norm_ratio'] - 1 + 1e-7):
                raise AssertionError('final representative bypasses a joint gate')
            for i, r in enumerate(mapping):
                if i != r and not (len(keys[r]) < len(keys[i]) and (keys[i].startswith(keys[r]) or keys[i].endswith(keys[r]))):
                    raise AssertionError('representative is not a proper prefix/suffix node')
        else:
            fidelity_before_quantization = None
        planning = {'seconds': time.perf_counter() - start, 'mapping_sha256': array_digest(mapping.astype('<u8')),
            'retained_rows': int(np.sum(mapping == np.arange(len(mapping)))), 'fidelity': fidelity_before_quantization}
    expected = None
    if job['backend'] == 'constant':
        if job['encoding'] == 'zero':
            vector = np.zeros(source.shape[1], dtype=np.float32)
        elif job['encoding'] == 'mean':
            vector = source.astype(np.float64).mean(0).astype(np.float32)
        else:
            vector = np.average(source.astype(np.float64), axis=0, weights=np.load(common / 'source-occurrence-counts.npy')).astype(np.float32)
        pack_quantized(vector[None], directory, 'fp32', native=False)
        expected = np.repeat(vector[None], len(source), axis=0)
    elif job['backend'] in ('native', 'raw', 'dense'):
        pack_quantized(source, directory, job['encoding'], native=job['backend'] == 'native',
            representatives=mapping.tolist() if job['planner'] in ('datrie', 'reference') else None,
            row_order=order, deduplicate=job['planner'] == 'byte-dedup', max_block_bytes=8192)
        expected = decode_rows(encode_rows(source[mapping], job['encoding']), source.shape[1], job['encoding'])
    elif job['backend'] == 'faiss':
        train_ids = np.load(common / 'fit-original-ids.npy')
        fit_codec(directory, source, job['encoding'], source[train_ids], seed=42, iterations=20, opq_iterations=10)
    else:
        original = Path(job['direct_directory']); directory.mkdir()
        reader = FaissRows(original)
        encoded = np.asarray(reader.codes).view('u1').reshape(len(reader.codes), -1)
        unique, code_mapping = np.unique(encoded, axis=0, return_inverse=True)
        dtype = next(t for t in (np.uint8, np.uint16, np.uint32, np.uint64) if len(unique) - 1 <= np.iinfo(t).max)
        np.save(directory / 'codes.npy', unique.view(reader.codes.dtype).reshape(-1, reader.codes.shape[1]))
        np.save(directory / 'original-to-code.npy', code_mapping.astype(dtype))
        m = load(original / 'codec.json'); m['rows'] = len(unique)
        write(directory / 'codec.json', m)
        if reader.index is not None:
            shutil.copyfile(original / 'codec.faiss', directory / 'codec.faiss')
        original_encoded_hash = array_digest(encoded)
        del reader
    construction_seconds = time.perf_counter() - started
    construction_memory = memory()
    with_reader = Reader(job)
    row_digest = hashlib.sha256(); squared = norm = 0.; maximum_decode_difference = 0.
    direct_reader = FaissRows(job['direct_directory']) if job['backend'] == 'dedup' else None
    for start in range(0, len(source), 4096):
        ids = np.arange(start, min(start + 4096, len(source)))
        decoded = with_reader.rows(with_reader.mapping[ids] if with_reader.mapping is not None else ids)
        if not np.isfinite(decoded).all():
            raise AssertionError('nonfinite reopened rows')
        if expected is not None and decoded.tobytes() != expected[ids].tobytes():
            raise AssertionError('native/raw output bytes differ from encoded final representatives')
        if direct_reader is not None:
            a = direct_reader.rows(ids)
            np.testing.assert_allclose(decoded, a, rtol=1e-6, atol=1e-7)
            maximum_decode_difference = max(maximum_decode_difference, float(np.abs(decoded - a).max()))
        row_digest.update(decoded.astype('<f4').tobytes())
        squared += float(np.square(source[ids].astype(np.float64) - decoded).sum())
        norm += float(np.square(source[ids].astype(np.float64)).sum())
    files = serving_files(directory)
    result = {'name': job['name'], 'family': job['family'], 'encoding': job['encoding'],
        'source_checkpoint_sha256': metadata['source_checkpoint_sha256'], 'job_sha256': digest(job_path),
        'environment': environment(), 'construction_seconds': construction_seconds,
        'construction_memory_before': before, 'construction_memory': construction_memory,
        'planning': planning, 'serving_files': files, 'row_bundle_bytes': sum(f['bytes'] for f in files.values()),
        'common_model_bytes': metadata['common_model_bytes'],
        'decoded_source_ids_verified': len(source), 'decoded_row_sha256': row_digest.hexdigest(),
        'stored_row_relative_frobenius': (squared / norm) ** .5, 'validation': {},
        'original_encoded_code_sha256': original_encoded_hash if job['backend'] == 'dedup' else None,
        'maximum_dedup_decode_difference': maximum_decode_difference if job['backend'] == 'dedup' else None,
        'cosine': job.get('cosine'), 'norm_ratio': job.get('norm_ratio')}
    result['model_bytes'] = result['row_bundle_bytes'] + result['common_model_bytes']
    for cached in (False, True):
        hidden = evaluate_hidden(with_reader, common, cached)
        quality, blocks = score_candidate(hidden, common, 'validation')
        result['validation'][str(int(cached))] = quality
        write(Path(job['report']).with_name(job['name'] + f'-validation-cache{int(cached)}-blocks.json.gz'), blocks)
    with_reader.close()
    result['status'] = 'complete'
    write(job['report'], result)


def query_job(job_path, cached, *, index_kind='datrie', cache_bytes=0, selected=False):
    import faiss
    faiss.omp_set_num_threads(1)
    from .benchmark_faiss_cotraining import evaluate, fidelity
    from .structural_native import StructuralIndex
    job = load(job_path); common = Path(job['common']); metadata = load(common / 'metadata.json')
    measured = {'environment': environment(), 'job_sha256': digest(job_path), 'cached_known_words': cached,
                'index_kind': index_kind, 'native_cache_bytes': cache_bytes, 'memory_before_open': memory()}
    start = time.perf_counter()
    reader = Reader(job, cache_bytes=cache_bytes)
    word_rows = np.load(common / ('cached-word-rows.npy' if cached else 'word-rows.npy'), mmap_mode='r')
    word_lookup = {w: i for i, w in enumerate(load(common / 'word-keys.json'))}
    index_path = common / ('keys.datrie' if index_kind == 'datrie' else 'keys.flat')
    structural_library = os.environ.get('MEFT_QUERY_STRUCTURAL_LIBRARY', job['structural_library'])
    index = StructuralIndex(index_path, structural_library)
    measured['structural_library_sha256'] = digest(structural_library)
    measured['load_seconds'] = time.perf_counter() - start
    inputs = load(common / 'query-strings.json')
    measured['query'] = {}; plans = {}
    for name, strings in inputs.items():
        bags, ids = lookup_strings(strings, index, word_lookup, metadata['order'], cached, index_kind)
        plans[name] = plan_for(bags, ids, reader)
        measured['query'][name] = {'inputs': len(strings), 'empty_executed_ngram_bags': sum(not b for b in bags),
            'known_cache_hits': sum(w >= 0 for w in ids) if cached else 0,
            'original_ngram_occurrences': sum(map(len, bags)), 'reducer_occurrences': len(plans[name]['ids']),
            'replay_samples_ms': [], 'complete_samples_ms': [], 'batch_complete_samples_ms': []}
    for _ in range(2):
        for name, plan in plans.items():
            evaluate(reader, plan, word_rows, batch_bags=64)
    rng = np.random.default_rng(42)
    for _ in range(15):
        for name in rng.permutation(list(plans)):
            start = time.perf_counter()
            output = evaluate(reader, plans[name], word_rows, batch_bags=64)
            measured['query'][name]['replay_samples_ms'].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter(); outputs = []
            for begin in range(0, len(inputs[name]), 64):
                batch_start = time.perf_counter()
                bags, ids = lookup_strings(inputs[name][begin:begin + 64], index, word_lookup, metadata['order'], cached, index_kind)
                outputs.append(evaluate(reader, plan_for(bags, ids, reader), word_rows, batch_bags=64))
                measured['query'][name]['batch_complete_samples_ms'].append((time.perf_counter() - batch_start) * 1000)
            measured['query'][name]['complete_samples_ms'].append((time.perf_counter() - start) * 1000)
            np.testing.assert_allclose(np.concatenate(outputs), output, rtol=1e-5, atol=1e-7)
    measured['memory_after_query'] = memory()
    for name, plan in plans.items():
        values = evaluate(reader, plan, word_rows, batch_bags=64)
        ref = np.load(common / (name + '-query-reference.npy'))
        q = measured['query'][name]
        q.update(complete_p50_ms=float(np.median(q['complete_samples_ms'])),
                 complete_p95_ms=float(np.quantile(q['complete_samples_ms'], .95)),
                 replay_p50_ms=float(np.median(q['replay_samples_ms'])),
                 batch_p95_ms=float(np.quantile(q['batch_complete_samples_ms'], .95)),
                 inputs_per_second=len(values) / (float(np.median(q['complete_samples_ms'])) / 1000),
                 output_sha256=array_digest(values), fidelity=fidelity(ref, values, np.ones(len(values), dtype=bool)))
    if not selected:
        measured['quality'] = {}
        hidden = evaluate_hidden(reader, common, cached)
        for split in ('test', 'confirmation'):
            quality, blocks = score_candidate(hidden, common, split)
            measured['quality'][split] = quality
            write(Path(job['report']).with_name(job['name'] + f'-{split}-cache{int(cached)}-blocks.json.gz'), blocks)
    reader.close(); index.close()
    measured['status'] = 'complete'
    suffix = f'-cache{int(cached)}-{index_kind}-rowcache{cache_bytes}'
    path = Path(job['report']).with_name(job['name'] + suffix + ('-retiming.json' if selected else '-audit.json'))
    write(path, measured)


def prepare_model(checkpoint_path, training, split, out, native_library, structural_library):
    import torch
    from .benchmark_faiss_cotraining import objective, evaluate
    from .structural_native import StructuralPlan
    from .faiss_row_codec import DEFAULT_CODECS
    from .train_ngram_similarity import ngrams
    preparation_started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    expected = next(r for r in training['runs'] if (r['convention'], r['order'], r['seed']) ==
        (checkpoint_path.parent.parent.name, checkpoint['order'], checkpoint['seed']))
    assert digest(checkpoint_path) == expected['checkpoint_sha256']
    common = out / 'common'; common.mkdir(parents=True)
    rows = checkpoint['input_rows'].numpy(); words, grams = checkpoint['words'], checkpoint['grams']
    source = np.ascontiguousarray(rows[len(words) + 1:])
    counts = Counter(split['train']); keys = words + sorted(set(w for name in SPLITS for w in split[name]) - set(words))
    lexical_ids = {w: i for i, w in enumerate(keys)}
    gram_ids = {g: i for i, g in enumerate(grams)}
    bags = [[gram_ids[g] for g in ngrams(w, checkpoint['order']) if g in gram_ids] for w in keys]
    word_ids = np.asarray([i if i < len(words) else -1 for i in range(len(keys))], dtype='<i8')
    np.save(common / 'source-rows.npy', source)
    occurrence = np.zeros(len(source), dtype=np.int64)
    for w, bag in zip(words, bags[:len(words)]):
        np.add.at(occurrence, bag, counts[w])
    np.save(common / 'source-occurrence-counts.npy', occurrence)
    np.save(common / 'word-rows.npy', rows[1:len(words) + 1])
    np.save(common / 'output-rows.npy', checkpoint['output_rows'].numpy())
    np.save(common / 'lexical-word-ids.npy', word_ids)
    np.save(common / 'lexical-nonempty.npy', np.asarray([bool(b) for b in bags]))
    write(common / 'word-keys.json', words); write(common / 'gram-keys.json', grams)
    write(common / 'lexical-bags.json', bags)
    index_started = time.perf_counter()
    with StructuralPlan(source.tobytes(), source.shape[1], dict(enumerate(grams)), structural_library, mode='layout') as plan:
        plan.save_index(common / 'keys.datrie'); plan.save_index(common / 'keys.flat', index='flat')
        assert plan.lookup(grams) == list(range(len(grams)))
    index_construction = {'seconds': time.perf_counter() - index_started, 'memory': memory(),
                          'all_original_ids_verified': len(grams)}
    class Oracle:
        metadata = {'rows': len(source), 'dim': source.shape[1]}
        mapping = None
        def rows(self, ids):
            return source[ids]
    oracle = Oracle(); hidden = evaluate(oracle, plan_for(bags, word_ids, oracle), rows[1:len(words) + 1], batch_bags=64)
    from .benchmark_heldout_compression import compose
    original_forward = compose(checkpoint['input_rows'], checkpoint['bags'], checkpoint['lengths']).numpy()
    original_forward_relative_l2 = float(np.linalg.norm(hidden[:len(words)].astype(np.float64) - original_forward) /
                                       np.linalg.norm(original_forward.astype(np.float64)))
    np.save(common / 'cached-word-rows.npy', hidden[:len(words)])
    np.save(common / 'lexical-source-hidden.npy', hidden)
    query = {}; rng = np.random.default_rng(42)
    for name, types in (('known', sorted(set(split['confirmation']) & set(words))),
                        ('oov', sorted(set(split['confirmation']) - set(words)))):
        picked = [types[i] for i in sorted(rng.choice(len(types), min(512, len(types)), replace=False))]
        query[name] = picked
        np.save(common / (name + '-query-reference.npy'), hidden[[lexical_ids[w] for w in picked]])
    write(common / 'query-strings.json', query)
    evaluations = {}
    for index, name in enumerate(SPLITS):
        arrays, info = context_evaluation(split[name], words, lexical_ids, counts, 42 + index)
        np.savez(common / (name + '-evaluation.npz'), **arrays)
        np.savez(common / (name + '-source-scores.npz'), **objective(hidden, checkpoint['output_rows'].numpy(), arrays))
        evaluations[name] = info
    ids = np.random.default_rng(42).choice(len(source), min(16384, len(source)), replace=False)
    np.save(common / 'fit-original-ids.npy', ids)
    file_names = ('keys.datrie', 'word-keys.json', 'word-rows.npy', 'output-rows.npy')
    metadata = {'source_checkpoint_sha256': digest(checkpoint_path), 'source_row_sha256': array_digest(source),
        'order': checkpoint['order'], 'seed': checkpoint['seed'], 'convention': expected['convention'],
        'shape': list(source.shape), 'evaluations': evaluations,
        'fit_original_ids_sha256': array_digest(ids.astype('<u8')),
        'common_serving_files': {name: {'bytes': (common / name).stat().st_size, 'sha256': digest(common / name)} for name in file_names},
        'index_bytes': {'datrie': (common / 'keys.datrie').stat().st_size, 'hash': (common / 'keys.flat').stat().st_size},
        'known_cache_file_bytes': (common / 'cached-word-rows.npy').stat().st_size,
        'canonical_source_reducer_vs_original_fp32_forward_relative_l2': original_forward_relative_l2,
        'common_index_construction': index_construction,
        'common_preparation_seconds': time.perf_counter() - preparation_started,
        'query_types': {name: len(v) for name, v in query.items()}}
    metadata['common_model_bytes'] = sum(v['bytes'] for v in metadata['common_serving_files'].values())
    assert metadata['known_cache_file_bytes'] == metadata['common_serving_files']['word-rows.npy']['bytes']
    write(common / 'metadata.json', metadata)
    jobs = []
    def add(name, family, backend, encoding, planner='none', **options):
        directory = out / 'bundles' / name; directory.parent.mkdir(exist_ok=True)
        report = out / 'reports' / (name + '.json'); report.parent.mkdir(exist_ok=True)
        job = {'name': name, 'family': family, 'backend': backend, 'encoding': encoding, 'planner': planner,
            'common': str(common), 'directory': str(directory), 'native_library': str(native_library),
            'structural_library': str(structural_library), 'report': str(report), **options}
        job_path = out / 'jobs' / (name + '.json'); write(job_path, job)
        jobs.append(str(job_path))
    for encoding in ('fp32', 'fp16', 'int8'):
        add('raw-' + encoding, 'raw', 'raw', encoding)
        for planner in ('none', 'layout', 'byte-dedup'):
            add('native-' + planner + '-' + encoding, 'native-identity', 'native', encoding, planner)
    add('dense-fp32', 'dense', 'dense', 'fp32')
    for encoding in ('zero', 'mean', 'weighted-mean'):
        add('constant-' + encoding, 'constant', 'constant', encoding)
    for planner in ('datrie', 'reference'):
        for cosine in GATES:
            for norm in GATES:
                for encoding in ('fp32', 'fp16', 'int8'):
                    add(f'tree-{planner}-c{cosine}-n{norm}-{encoding}', 'tree-' + planner, 'native', encoding,
                        planner, cosine=cosine, norm_ratio=norm)
    for encoding in DEFAULT_CODECS:
        add('faiss-' + encoding, 'faiss', 'faiss', encoding)
        add('dedup-' + encoding, 'faiss-dedup', 'dedup', encoding,
            direct_directory=str(out / 'bundles' / ('faiss-' + encoding)))
    assert len(jobs) == 204
    write(out / 'jobs.json', jobs)
    return {'name': out.name, 'metadata_sha256': digest(common / 'metadata.json'), 'jobs': jobs}


def execute_jobs(jobs, phase, workers, root):
    report = {'phase': phase, 'jobs': len(jobs), 'complete': [], 'failures': []}
    write(root / (phase + '-progress.json'), report)
    def execute(path, cached=None):
        job = load(path); log = Path(job['report']).with_suffix(f'.{phase}-{cached}.log')
        command = [sys.executable, '-m', 'tools.tree_node_quality', phase, '--job', path]
        if cached is not None:
            command += ['--cached', str(cached)]
        with log.open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{job["name"]} {phase} {cached}: exit {result.returncode}; {log}')
        return {'job_sha256': digest(path), 'name': job['name'], 'model': Path(job['common']).parent.name, 'cached': cached}
    requested = [(path, cached) for path in jobs for cached in ((None,) if phase == 'build' else (0, 1))]
    np.random.default_rng(42).shuffle(requested)
    report['execution_schedule_sha256'] = hashlib.sha256(json.dumps(requested).encode()).hexdigest()
    write(root / (phase + '-progress.json'), report)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(execute, path, cached): (path, cached) for path, cached in requested}
        for future in as_completed(pending):
            try:
                report['complete'].append(future.result())
            except Exception as error:
                report['failures'].append({'job': pending[future], 'error': repr(error)})
            write(root / (phase + '-progress.json'), report)
            if len(report['complete']) % 20 == 0 or report['failures']:
                print(json.dumps({'phase': phase, 'complete': len(report['complete']), 'expected': len(requested), 'failures': len(report['failures'])}), flush=True)
    if report['failures'] or len(report['complete']) != len(requested):
        raise RuntimeError('complete phase required; preserve failures')


def freeze_selection(root):
    selections = {'environment': environment(), 'selection_uses': 'validation only', 'models': []}
    for model in load(root / 'prepare.json')['models']:
        candidates = [load(load(path)['report']) for path in model['jobs']]
        choices = []
        for cached in (0, 1):
            for budget in BUDGETS:
                for family in ('raw', 'dense', 'constant', 'native-identity', 'tree-datrie', 'tree-reference', 'faiss', 'faiss-dedup'):
                    chosen = select([c for c in candidates if c['family'] == family], budget, cached)
                    choices.append({'cached': cached, 'budget': budget, 'family': family,
                        'name': chosen['name'] if chosen else None, 'model_bytes': chosen['model_bytes'] if chosen else None})
        selections['models'].append({'name': model['name'], 'choices': choices,
            'candidate_report_sha256': {c['name']: digest(load(path)['report']) for c, path in zip(candidates, model['jobs'])}})
    write(root / 'selection.json', selections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'build-grid', 'audit-grid', 'build', 'audit'))
    parser.add_argument('--training-root', type=Path)
    parser.add_argument('--corpus-zip', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--structural-library', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--cached', type=int, choices=(0, 1), default=0)
    parser.add_argument('--jobs', type=int, default=9)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU only; empty CUDA_VISIBLE_DEVICES required')
    if args.phase == 'build':
        build_job(args.job); return
    if args.phase == 'audit':
        query_job(args.job, args.cached); return
    root = args.output.resolve()
    if args.phase == 'prepare':
        import torch
        from .benchmark_heldout_compression import split_tokens
        from .train_corpus_checkpoints import corpus_prefix
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        training = load(args.training_root / 'report.json')
        assert training['status'] == 'completed'
        assert digest(args.corpus_zip) == training['corpus_zip_sha256']
        original, _ = corpus_prefix(args.corpus_zip, 'enwik8', 200000, 4 * 1024**2)
        full, raw_hash = corpus_prefix(args.corpus_zip, 'enwik8', 300000, 4 * 1024**2)
        assert digest(args.training_root / 'corpus.txt') == training['tokenized_corpus_sha256']
        assert original == (args.training_root / 'corpus.txt').read_text()
        assert ' '.join(full.split()[:200000]) == original
        split = split_tokens(original); split['confirmation'] = full.split()[200000:300000]
        root.mkdir(parents=True, exist_ok=False)
        report = {'environment': environment(), 'torch': torch.__version__,
            'source_training_report_sha256': digest(args.training_root / 'report.json'),
            'corpus_zip_sha256': digest(args.corpus_zip), 'raw_prefix_sha256': raw_hash,
            'confirmation_token_range': [200000, 300000],
            'confirmation_tokens_sha256': array_digest(np.frombuffer(' '.join(split['confirmation']).encode(), dtype='u1')),
            'models': [], 'native_library_sha256': digest(args.native_library),
            'structural_library_sha256': digest(args.structural_library)}
        for convention in ('legacy', 'average'):
            for order in (4, 8, 12):
                for seed in (3, 7, 11):
                    name = f'{convention}-order-{order}-seed-{seed}'
                    report['models'].append(prepare_model(args.training_root / convention / f'order-{order}-seed-{seed}' / 'reference.pt',
                        training, split, root / name, args.native_library, args.structural_library))
                    write(root / 'prepare.json', report)
                    print(json.dumps({'prepared': name}), flush=True)
        report['status'] = 'complete'; write(root / 'prepare.json', report)
        return
    prepare = load(root / 'prepare.json'); assert prepare['status'] == 'complete'
    jobs = [path for model in prepare['models'] for path in model['jobs']]
    if args.phase == 'build-grid':
        base = [path for path in jobs if load(path)['backend'] != 'dedup']
        dedup = [path for path in jobs if load(path)['backend'] == 'dedup']
        execute_jobs(base, 'build', args.jobs, root)
        execute_jobs(dedup, 'build', args.jobs, root)
        freeze_selection(root)
    else:
        selection_hash = digest(root / 'selection.json')
        execute_jobs(jobs, 'audit', args.jobs, root)
        assert digest(root / 'selection.json') == selection_hash
        write(root / 'complete.json', {'status': 'complete', 'models': 18, 'requested_candidates': len(jobs),
            'fresh_audit_workers': len(jobs) * 2, 'frozen_selection_sha256': selection_hash})


if __name__ == '__main__':
    main()
