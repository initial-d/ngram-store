"""Frozen-source codec frontier and simultaneous calibration for large tables.

All construction uses trained rows. Calibration selects; official test only
confirms. Identical decoded representations share a quality result but receive
separate actual file-capacity and query measurements.
"""
import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .faiss_row_codec import DEFAULT_CODECS, FaissRows, fit_codec, pack_codec_native
from .large_ngram_application import batch_csr
from .native_row_store import NativeRows, pack_native
from .tree_node_quality import array_digest, digest, environment, load, memory, serving_files, write


CODECS = DEFAULT_CODECS


def publish_job(path, job):
    """Count the actual required reader metadata, including this job file."""
    base = job['complete_model_bytes']
    size = 0
    while True:
        job['required_job_metadata_bytes'] = size
        job['complete_model_bytes'] = base + size
        write(path, job)
        actual = path.stat().st_size
        if actual == size: return
        size = actual


def select_sources(args):
    groups = defaultdict(list)
    reports = sorted(args.training.glob('*/training.json'))
    if len(reports) != 144:
        raise ValueError('complete registered 144-cell training grid required')
    unqualified = []
    for p in reports:
        r = load(p)
        if r['status'] == 'incomplete-row-coverage':
            if len(r['epochs']) != 10 or not (p.parent / 'rows.npy').is_file() or not (p.parent / 'classifier.npz').is_file():
                raise ValueError('incomplete failed training endpoint')
            unqualified.append({'training': str(p.parent), 'report_sha256': digest(p),
                'reason': 'Registered requirement that every source row change from initialization failed.',
                'changed_rows': r['changed_rows'], 'registered_nonpadding_rows': r['rows'] - 1,
                'rows_sha256': digest(p.parent / 'rows.npy')})
        elif r['status'] != 'complete' or not r.get('calibration'):
            raise ValueError('complete trained rows and calibration required: ' + str(p))
        elif digest(p.parent / 'rows.npy') != r['rows_sha256'] or digest(p.parent / 'calibration.npz') != r['calibration']['sha256']:
            raise ValueError('trained fixture changed')
        s = r['settings']
        dataset = Path(s['data']).name
        family = (dataset, r['initialization'], r['addressing'], s['buckets'] if r['addressing'] == 'hash-budget' else 0)
        groups[family].append((p, r))
    expected_families = {(dataset, init, addressing, buckets)
        for dataset in ('ag_news', 'dbpedia_14') for init in ('zero', 'uniform')
        for addressing, buckets in (('exact', 0), ('hash-matched', 0),
                                   ('hash-budget', 1 << 20), ('hash-budget', 1 << 22))}
    if set(groups) != expected_families:
        raise ValueError('registered dataset/initialization/addressing families changed')
    chosen = []
    for family, cells in sorted(groups.items()):
        by_rate = defaultdict(list)
        for p, r in cells: by_rate[r['settings']['lr']].append((p, r))
        if sorted(by_rate) != [1., 3., 10.] or any(sorted(r['seed'] for _, r in x) != [3, 7, 11] for x in by_rate.values()):
            raise ValueError('incomplete rate/seed family')
        eligible = {lr: cells for lr, cells in by_rate.items() if all(r['status'] == 'complete' for _, r in cells)}
        if not eligible:
            raise ValueError('no rate meets the preregistered complete-row requirement: ' + str(family))
        rate = min(eligible, key=lambda lr: (np.mean([r['calibration']['mean_nll'] for _, r in eligible[lr]]), lr))
        for p, r in by_rate[rate]:
            chosen.append({'family': list(family), 'seed': r['seed'], 'learning_rate': rate,
                'training': str(p.parent), 'report_sha256': digest(p), 'calibration': r['calibration']})
    write(args.output, {'status': 'frozen', 'rule': 'Mean calibration NLL across all three seeds; minimum within addressing/initialization/dataset only.',
        'test_used': False, 'source_sha256': digest(__file__), 'selected': chosen,
        'training_grid': {str(p): digest(p) for p in reports}, 'unqualified_training_cells': unqualified,
        'eligibility': 'All three seeds must meet the predeclared row-coverage requirement before a learning rate can qualify. Failed cells remain in the complete grid.'})


def decoded_digest(reader, rows):
    sha = __import__('hashlib').sha256()
    for start in range(0, rows, 8192):
        ids = np.arange(start, min(start + 8192, rows), dtype=np.int64)
        data = reader.rows(ids)
        if data.dtype != np.float32 or not np.isfinite(data).all():
            raise ValueError('invalid actual decoded rows')
        sha.update(np.ascontiguousarray(data, dtype='<f4').tobytes())
    return sha.hexdigest()


def build(args):
    import faiss
    faiss.omp_set_num_threads(args.threads)
    selected = load(args.selection)
    source = args.training
    registration = next((e for e in selected['selected'] if e['training'] == str(source)), None)
    if registration is None or registration['family'][2] != 'exact':
        raise ValueError('only frozen validation-selected full-key sources enter this storage frontier')
    r = load(source / 'training.json'); data = Path(r['settings']['data'])
    rows = np.load(source / 'rows.npy', mmap_mode='r')
    if digest(source / 'rows.npy') != r['rows_sha256']:
        raise ValueError('trained original rows changed')
    args.output.mkdir(parents=True, exist_ok=True)
    # Fit on actual training-derived rows, independent of calibration/test keys.
    rng = np.random.default_rng(42)
    fit_ids = np.sort(rng.choice(np.arange(1, len(rows)), min(65536, len(rows) - 1), replace=False))
    np.save(args.output / 'fit-ids.npy', fit_ids)
    shared = {name: {'bytes': (data / name).stat().st_size, 'sha256': digest(data / name)}
              for name in ('word-vocabulary.json', 'registry.npy', 'prepared.json')}
    shared['classifier.npz'] = {'bytes': (source / 'classifier.npz').stat().st_size, 'sha256': digest(source / 'classifier.npz')}
    common_bytes = sum(v['bytes'] for v in shared.values())
    jobs = []
    for codec in CODECS:
        direct = args.output / 'bundles' / ('faiss-' + codec)
        if not (direct / 'codec.json').exists():
            fit_codec(direct, rows, codec, rows[fit_ids], seed=42, iterations=20)
        with FaissRows(direct) as reader:
            sha = decoded_digest(reader, len(rows))
        entries = [('direct-' + codec, direct, sha, 'faiss')]
        for block_codec in ('zlib', 'lz4', 'zstd'):
            name = f'native-{block_codec}-{codec}'
            directory = args.output / 'bundles' / name
            if not (directory / 'codec.json').exists():
                pack_codec_native(direct, directory, native_library=args.library,
                    max_block_bytes=8192, block_rows=256, predictor='xor', block_codec=block_codec)
            with FaissRows(directory, native_library=args.library) as reader:
                if decoded_digest(reader, len(rows)) != sha:
                    raise ValueError('composition changed codebook/decoded original rows')
            entries.append((name, directory, sha, 'faiss'))
        for name, directory, sha, kind in entries:
            files = serving_files(directory)
            job = {'name': name, 'directory': str(directory), 'kind': kind, 'source': str(source),
                'data': str(data), 'source_report_sha256': digest(source / 'training.json'), 'rows': len(rows),
                'dim': rows.shape[1], 'decoded_sha256': sha, 'library': str(args.library),
                'library_sha256': digest(args.library), 'common_files': shared,
                'complete_model_bytes': common_bytes + sum(v['bytes'] for v in files.values()),
                'files': files, 'quality_reference': 'direct-' + codec, 'source_sha256': digest(__file__)}
            path = args.output / 'jobs' / (name + '.json'); publish_job(path, job); jobs.append(str(path))
    # Independent generic-byte baselines use the same C gather and descriptors,
    # but no XOR, aliases or structural row grouping. This controls reader cost.
    raw = args.output / 'source.raw'
    if not raw.exists():
        with raw.open('wb') as f:
            for begin in range(0, len(rows), 8192): rows[begin:begin + 8192].tofile(f)
    for block_codec in ('zlib', 'lz4', 'zstd'):
        for sharing in ('generic', 'byte-dedup'):
            name = sharing + '-' + block_codec; directory = args.output / 'bundles' / name
            directory.mkdir(parents=True, exist_ok=True)
            artifact = directory / 'rows.meft'
            if not artifact.exists():
                pack_native(raw, artifact, rows.shape[1] * 4, 256,
                    deduplicate=sharing == 'byte-dedup', predictor='none', codec=block_codec)
            with NativeRows(artifact, args.library) as reader:
                h = __import__('hashlib').sha256()
                for begin in range(0, len(rows), 8192):
                    end = min(begin + 8192, len(rows)); actual = reader.rows(range(begin, end))
                    if actual != rows[begin:end].tobytes(): raise ValueError('generic compression changed source bytes')
                    h.update(actual)
                sha = h.hexdigest()
            files = serving_files(directory)
            job = {'name': name, 'directory': str(directory), 'kind': 'native', 'source': str(source),
                'data': str(data), 'source_report_sha256': digest(source / 'training.json'), 'rows': len(rows),
                'dim': rows.shape[1], 'decoded_sha256': sha, 'library': str(args.library),
                'library_sha256': digest(args.library), 'common_files': shared,
                'complete_model_bytes': common_bytes + sum(v['bytes'] for v in files.values()),
                'files': files, 'quality_reference': 'direct-fp32', 'source_sha256': digest(__file__)}
            path = args.output / 'jobs' / (name + '.json'); publish_job(path, job); jobs.append(str(path))
    if args.structural_library:
        from .structural_native import StructuralPlan
        registry = np.load(data / 'registry.npy', mmap_mode='r')
        order = load(data / 'prepared.json')['ngram_order']
        tokens = registry.view('<u4').reshape(-1, order)
        # Fixed-width delimited positive token IDs preserve exact containment;
        # padding zeros are omitted. They are canonical keys, not hash buckets.
        keys = {i + 1: ''.join(f'{int(v):08x}|' for v in token if v) for i, token in enumerate(tokens)}
        for mode, gate in [('exact', 1.), *[('joint', t) for t in (.99, .98, .97, .95, .90)]]:
            name = f'datrie-{mode}-{gate:g}'; directory = args.output / 'bundles' / name
            directory.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter(); before = memory()
            with StructuralPlan.from_array(rows, keys, args.structural_library, mode=mode, cosine=gate, norm_ratio=.99) as plan:
                artifact = directory / 'rows.meft'
                if not artifact.exists():
                    storage = pack_native(raw, artifact, rows.shape[1] * 4, 256, False, 'xor',
                        representatives=plan.representatives, row_order=plan.row_order)
                with NativeRows(artifact, args.library) as reader:
                    sha = __import__('hashlib').sha256()
                    for begin in range(0, len(rows), 8192):
                        end = min(begin + 8192, len(rows)); actual = reader.rows(range(begin, end))
                        expected = rows[plan.representatives[begin:end]].tobytes()
                        if actual != expected: raise ValueError('DATrie mark/compact changed representative bytes')
                        sha.update(actual)
                    if sha.hexdigest() != plan.reconstructed_sha256: raise ValueError('structural reconstructed digest changed')
                structural = {'mode': mode, 'cosine': gate, 'norm_ratio': .99,
                    'retained_rows': sum(i == r for i, r in enumerate(plan.representatives)),
                    'original_bytes_preserved': plan.original_bytes_preserved,
                    'source_sha256': plan.source_sha256, 'reconstructed_sha256': plan.reconstructed_sha256,
                    'planner_library_sha256': digest(args.structural_library), 'borrowed_readonly_rows': True,
                    'construction_seconds': time.perf_counter() - start, 'memory_before': before, 'memory_after': memory(),
                    'key_contract': 'Training-only fixed-width token fields; retained proper prefix/suffix containment checked by the original DATrie callbacks and final direct cosine/norm guard.'}
                write(directory / 'structural.json', structural)
                sha = plan.reconstructed_sha256
            files = serving_files(directory)
            job = {'name': name, 'directory': str(directory), 'kind': 'native', 'source': str(source),
                'data': str(data), 'source_report_sha256': digest(source / 'training.json'), 'rows': len(rows),
                'dim': rows.shape[1], 'decoded_sha256': sha, 'library': str(args.library),
                'library_sha256': digest(args.library), 'common_files': shared,
                'complete_model_bytes': common_bytes + sum(v['bytes'] for v in files.values()),
                'files': files, 'quality_reference': name, 'source_sha256': digest(__file__)}
            path = args.output / 'jobs' / (name + '.json'); publish_job(path, job); jobs.append(str(path))
    write(args.output / 'built.json', {'status': 'complete', 'jobs': jobs, 'selection_sha256': digest(args.selection),
        'fit_ids_sha256': digest(args.output / 'fit-ids.npy'), 'source_sha256': digest(__file__)})


class ApplicationRows:
    def __init__(self, job):
        self.job = job
        if job['kind'] == 'source':
            self.reader = np.load(Path(job['source']) / 'rows.npy', mmap_mode='r')
        else:
            self.reader = FaissRows(job['directory'], native_library=job['library']) if job['kind'] == 'faiss' else NativeRows(Path(job['directory']) / 'rows.meft', job['library'])

    def rows(self, ids):
        if self.job['kind'] == 'source': return np.ascontiguousarray(self.reader[ids])
        if self.job['kind'] == 'faiss': return self.reader.rows(ids)
        # The strict dependency-free native binding accepts Python integer IDs.
        # Preserve values/types (including rejection of floats/bools) rather
        # than silently casting the caller's NumPy scalars to integers.
        data = self.reader.rows(ids.tolist() if isinstance(ids, np.ndarray) else ids)
        return np.frombuffer(data, dtype='<f4').reshape(len(ids), self.job['dim'])

    def close(self):
        if self.job['kind'] == 'source': self.reader._mmap.close()
        else: self.reader.close()


def bag_logits(reader, classifier, ids, ptr, documents, mapping=None):
    """Identical FP64 reductions for every source/codec, excluding OOV zero IDs."""
    flat, offsets = batch_csr(ids, ptr, documents, mapping)
    dim = classifier['weight'].shape[1]
    hidden = np.zeros((len(documents), dim), dtype=np.float64)
    nonempty = np.diff(offsets) > 0
    if len(flat):
        values = reader.rows(flat).astype(np.float64)
        values[flat == 0] = 0
        starts = offsets[:-1][nonempty]
        summed = np.add.reduceat(values, starts, axis=0)
        counts = np.add.reduceat((flat != 0).astype(np.int64), starts)
        hidden[nonempty] = summed / np.maximum(counts[:, None], 1)
    return (hidden @ classifier['weight'].astype(np.float64).T + classifier['bias'].astype(np.float64)).astype(np.float32)


def evaluate(args):
    job = load(args.job); data = Path(job['data']); source = Path(job['source'])
    if digest(source / 'training.json') != job['source_report_sha256']:
        raise ValueError('source training changed')
    reader = ApplicationRows(job)
    classifier = np.load(source / 'classifier.npz')
    ids = np.load(data / (args.split + '-ids.npy'), mmap_mode='r')
    ptr = np.load(data / (args.split + '-offsets.npy'), mmap_mode='r')
    labels = np.load(data / (args.split + '-labels.npy'))
    map_path = source / 'canonical-to-bucket.npy'
    mapping = np.load(map_path, mmap_mode='r') if map_path.exists() else None
    report = {'status': 'running', 'job_sha256': digest(args.job), 'environment': environment(),
        'split': args.split, 'source_sha256': digest(__file__), 'rss_before': memory(),
        'reduction': 'FP64 mean of original addressed rows excluding zero OOV; same schedule for every representation'}
    started = time.perf_counter()
    try:
        logits = np.empty((len(labels), classifier['weight'].shape[0]), dtype=np.float32)
        for begin in range(0, len(labels), 256):
            docs = np.arange(begin, min(begin + 256, len(labels)))
            logits[docs] = bag_logits(reader, classifier, ids, ptr, docs, mapping)
        shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
        nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
        payload = args.output.with_suffix('.npz'); payload.parent.mkdir(parents=True, exist_ok=True)
        np.savez(payload, logits=logits, labels=labels, nll=nll)
        report.update(status='complete', documents=len(labels), accuracy=float(np.mean(logits.argmax(1) == labels)),
            mean_nll=float(nll.mean()), seconds=time.perf_counter() - started, payload_sha256=digest(payload),
            logits_sha256=array_digest(logits), rss_after=memory())
    except BaseException as e:
        report.update(status='failed', error=repr(e)); raise
    finally:
        reader.close(); classifier.close(); write(args.output, report)


def simultaneous_upper(delta, choices, alpha=.05):
    """Empirical Bernstein + union bound for bounded paired accuracy differences.

    Conditional on the fitted models, documents are assumed independent draws.
    Range is [-1,1]. This is a calibration population guarantee under that
    sampling assumption; it is not a universal OOD guarantee or seed-level CI.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if len(delta) < 2 or np.any(np.abs(delta) > 1): raise ValueError('invalid paired accuracy differences')
    log = math.log(2 * choices / alpha)
    return min(1., float(delta.mean() + np.sqrt(2 * delta.var(ddof=1) * log / len(delta)) + 14 * log / (3 * (len(delta) - 1))))


def bernoulli_upper(count, total, choices, alpha=.05):
    """Invert the one-sided binomial Chernoff KL bound, without SciPy."""
    if not 0 <= count <= total or total < 1 or choices < 1:
        raise ValueError('invalid harmful-disagreement counts')
    bound = math.log(choices / alpha) / total
    q = count / total
    if count == 0: return -math.expm1(-bound)
    if count == total: return 1.
    lo, hi = q, 1.
    for _ in range(64):
        p = (lo + hi) / 2
        kl = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
        if kl > bound: hi = p
        else: lo = p
    return hi


def stratified_harm_upper(labels, baseline_correct, candidate_correct, choices):
    """Safe accuracy-loss bound under the actual fixed-label stratified split.

    Accuracy loss is at most harmful disagreement probability. A union bound
    covers every model/codec and label stratum. Beneficial changes are ignored
    by this conservative gate; pointwise paired loss is still reported.
    """
    labels = np.asarray(labels); classes = np.unique(labels)
    harmful = baseline_correct & ~candidate_correct
    rows = []
    for label in classes:
        mask = labels == label; n = int(mask.sum()); k = int(harmful[mask].sum())
        rows.append({'label': int(label), 'documents': n, 'harmful': k,
            'upper': bernoulli_upper(k, n, choices * len(classes))})
    return sum(row['documents'] / len(labels) * row['upper'] for row in rows), rows


def freeze(args):
    built = load(args.built); jobs = [load(p) for p in built['jobs']]
    reference = np.load(args.results / 'direct-fp32-calibration.npz')
    labels = reference['labels']; base = reference['logits'].argmax(1) == labels
    candidates = []
    for job in jobs:
        result = load(args.results / (job['quality_reference'] + '-calibration.json'))
        payload = np.load(args.results / (job['quality_reference'] + '-calibration.npz'))
        if not np.array_equal(payload['labels'], labels): raise ValueError('calibration pairs changed')
        correct = payload['logits'].argmax(1) == labels
        exact = job['decoded_sha256'] == next(j['decoded_sha256'] for j in jobs if j['name'] == 'direct-fp32')
        delta = base.astype(np.float64) - correct.astype(np.float64)
        upper, class_bounds = (0., []) if exact else stratified_harm_upper(labels, base, correct, len(jobs) * 36)
        candidates.append({'name': job['name'], 'model_bytes': job['complete_model_bytes'],
            'accuracy_loss': float(delta.mean()), 'simultaneous_upper_accuracy_loss': upper,
            'stratified_harmful_bounds': class_bounds,
            'exact_source_rows': exact, 'calibration_report_sha256': digest(args.results / (job['quality_reference'] + '-calibration.json'))})
        payload.close()
    selected = {}
    for budget in (0., .001, .005, .01, .02):
        feasible = [c for c in candidates if c['simultaneous_upper_accuracy_loss'] <= budget]
        selected[str(budget)] = min(feasible, key=lambda c: (c['model_bytes'], c['name'])) if feasible else None
    write(args.output, {'status': 'frozen', 'built_sha256': digest(args.built), 'test_used': False,
        'selection': selected, 'candidates': candidates,
        'confidence_contract': 'Familywise 95% one-sided Chernoff-KL union bound on harmful prediction changes within each label stratum, under independent class-conditional calibration documents. Fixed-label stratification is respected. Source-byte-exact candidates have deterministic zero row loss.',
        'independent_seeds': 'Freeze each seed separately; report all three held-out outcomes and seed-cluster intervals separately.',
        'familywise_choices': len(jobs) * 36,
        'choice_correction': 'Includes 12 source seed/dataset/initialization families and all three possible source learning rates, before codec selection.',
        'source_sha256': digest(__file__)})
    reference.close()


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='phase', required=True)
    c = sub.add_parser('select-sources'); c.add_argument('--training', type=Path, required=True); c.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('build')
    for name in ('training', 'selection', 'output', 'library'): c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--threads', type=int, default=4)
    c.add_argument('--structural-library', type=Path, required=True)
    c = sub.add_parser('evaluate')
    for name in ('job', 'output'): c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--split', choices=('calibration', 'test'), required=True)
    c = sub.add_parser('freeze')
    for name in ('built', 'results', 'output'): c.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    {'select-sources': select_sources, 'build': build, 'evaluate': evaluate, 'freeze': freeze}[args.phase](args)


if __name__ == '__main__': main()
