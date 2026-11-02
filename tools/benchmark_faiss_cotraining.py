"""CPU same-task FAISS codec / incidence-group comparison on trained SGNS.

No ANN queries: exact original-ID gathering followed by weighted bag means.
Codebooks are fit ONLY to training-checkpoint rows. Validation selects storage
under fixed quality budgets; test is audited afterward. An explicit full-domain
hybrid stores original nonsingleton rows for rejected queries, counting all of
them even when the sampled OOV trace touches few rows.
"""
import argparse
from collections import Counter
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

from .cotraining_groups import BagPartition, _count_bags
from .faiss_row_codec import DEFAULT_CODECS, FaissRows, fit_codec


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def schedule(bags, word_ids, partition=None, *, fallback=False, code_mapping=None):
    """Deduplicate repeated IDs in ALL controls; only certified groups are tied."""
    if len(bags) != len(word_ids):
        raise ValueError('one word ID per bag required')
    if partition is not None and code_mapping is not None:
        raise ValueError('incidence and encoded-byte maps are separate controls')
    if partition is None:
        original = _count_bags(bags, nrows=None if code_mapping is None else len(code_mapping))
        bag_ids, ids, weights = (original[k] for k in ('bags', 'rows', 'counts'))
        if code_mapping is not None:
            ids = np.asarray(code_mapping)[ids]
        accepted = np.ones(len(bags), dtype=bool)
    else:
        compiled = partition._compiled_groups(bags)
        original = compiled['original']
        safe = compiled['accepted']
        used = safe[compiled['bags']]
        bag_ids, ids, weights = (compiled[k][used] for k in ('bags', 'groups', 'counts'))
        accepted = safe | fallback
        if fallback and not safe.all():
            multi = partition.sizes[partition.mapping] > 1
            rank = np.cumsum(multi, dtype=np.int64) - 1
            rejected = ~safe[original['bags']]
            rows = original['rows'][rejected]
            fallback_ids = np.where(multi[rows], len(partition.sizes) + rank[rows], partition.mapping[rows])
            bag_ids = np.r_[bag_ids, original['bags'][rejected]]
            ids = np.r_[ids, fallback_ids]
            weights = np.r_[weights, original['counts'][rejected]]
    # Apply the same batch counting optimization to the direct codec controls.
    reorder = code_mapping is not None or (partition is not None and fallback and not safe.all())
    if reorder:
        order = np.lexsort((ids, bag_ids))
        ids, bag_ids, weights = ids[order], bag_ids[order], weights[order]
    if code_mapping is not None and len(ids):
        starts = np.r_[0, np.flatnonzero((ids[1:] != ids[:-1]) | (bag_ids[1:] != bag_ids[:-1])) + 1]
        weights = np.add.reduceat(weights, starts)
        ids, bag_ids = ids[starts], bag_ids[starts]
    per_bag = np.bincount(bag_ids, minlength=len(bags))
    offsets = np.r_[0, np.cumsum(per_bag, dtype=np.int64)]
    denominators = np.maximum(1, original['lengths'] + (np.asarray(word_ids) >= 0))
    return {'ids': np.asarray(ids, dtype='<i8'), 'weights': np.asarray(weights, dtype='<f8'),
            'offsets': np.asarray(offsets, dtype='<i8'),
            'denominators': np.asarray(denominators, dtype='<f8'),
            'words': np.asarray(word_ids, dtype='<i8'), 'accepted': np.asarray(accepted, dtype=bool)}


def evaluate(reader, plan, word_rows, fallback_rows=None, batch_bags=64):
    result = np.zeros((len(plan['words']), reader.metadata['dim']), dtype=np.float32)
    base_count = reader.metadata['rows']
    for begin in range(0, len(result), batch_bags):
        end = min(begin + batch_bags, len(result))
        lo, hi = int(plan['offsets'][begin]), int(plan['offsets'][end])
        ids = plan['ids'][lo:hi]
        if fallback_rows is None:
            decoded = reader.rows(ids)
        else:
            normal = ids < base_count
            decoded = np.empty((len(ids), reader.metadata['dim']), dtype=np.float32)
            decoded[normal] = reader.rows(ids[normal])
            decoded[~normal] = fallback_rows[ids[~normal] - base_count]
        weighted = decoded.astype(np.float64) * plan['weights'][lo:hi, None]
        starts = plan['offsets'][begin:end] - lo
        nonempty = np.diff(plan['offsets'][begin:end + 1]) > 0
        sums = np.zeros((end - begin, reader.metadata['dim']), dtype=np.float64)
        if np.any(nonempty):
            sums[nonempty] = np.add.reduceat(weighted, starts[nonempty], axis=0)
        known = plan['words'][begin:end] >= 0
        sums[known] += word_rows[plan['words'][begin:end][known]]
        result[begin:end] = sums / plan['denominators'][begin:end, None]
    return result


def fidelity(reference, candidate, accepted):
    a, b = reference[accepted].astype(np.float64), candidate[accepted].astype(np.float64)
    if not len(a):
        return {'accepted': 0, 'total': len(accepted)}
    norm = np.linalg.norm(a, axis=1)
    error = np.linalg.norm(a - b, axis=1) / np.maximum(norm, 1e-30)
    cosine = (a * b).sum(1) / np.maximum(norm * np.linalg.norm(b, axis=1), 1e-30)
    return {'accepted': len(a), 'total': len(accepted),
            'global_relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)),
            'relative_l2_p50': float(np.median(error)), 'relative_l2_p95': float(np.quantile(error, .95)),
            'relative_l2_max': float(error.max()), 'cosine_mean': float(cosine.mean())}


def objective(hidden, outputs, evaluation):
    losses, wins = [], []
    for begin in range(0, len(evaluation['pairs']), 1024):
        pairs = evaluation['pairs'][begin:begin + 1024]
        negatives = evaluation['negative'][begin:begin + 1024]
        h = hidden[pairs[:, 0]].astype(np.float64)
        positive = (h * outputs[pairs[:, 1]]).sum(1)
        sampled = (h[:, None] * outputs[negatives]).sum(2)
        losses.append(np.logaddexp(0, -positive) + np.logaddexp(0, sampled).sum(1))
        wins.append(((positive[:, None] > sampled) + .5 * (positive[:, None] == sampled)).mean(1))
    return {'loss': np.concatenate(losses), 'discrimination': np.concatenate(wins)}


def quality_summary(candidate, baseline):
    return {'sgns_loss_per_pair': float(candidate['loss'].mean()),
            'context_discrimination': float(candidate['discrimination'].mean()),
            'loss_relative_change': float(candidate['loss'].mean() / baseline['loss'].mean() - 1),
            'discrimination_change': float(candidate['discrimination'].mean() - baseline['discrimination'].mean())}


def worker(job_path):
    import faiss
    faiss.omp_set_num_threads(1)
    job = json.loads(Path(job_path).read_text())
    common = Path(job['common'])
    start = time.perf_counter()
    reader = FaissRows(job['codec_dir'])
    words = np.load(common / 'word-rows.npy', mmap_mode='r')
    fallback = np.load(common / 'fallback.npy', mmap_mode='r') if job['fallback'] else None
    partition = BagPartition.load(common / 'domain.npz') if job['grouped'] else None
    code_mapping = np.load(job['code_mapping'], mmap_mode='r') if job.get('code_mapping') else None
    original_traces = json.loads((common / 'trace-bags.json').read_text())
    traces = {name: dict(np.load(path, allow_pickle=False)) for name, path in job['trace_plans'].items()}
    result = {'name': job['name'], 'codec': reader.metadata, 'coverage': job['coverage'],
              'model_bytes': job['model_bytes'], 'fixed_trace_plan_bytes': job['fixed_trace_plan_bytes'],
              'load_seconds': time.perf_counter() - start, 'query': {}}
    rng = np.random.default_rng(42)
    samples = {name: [] for name in traces}
    for _ in range(2):
        for name, plan in traces.items():
            evaluate(reader, plan, words, fallback, job['batch_bags'])
    for _ in range(job['repeat']):
        for name in rng.permutation(list(traces)):
            begin = time.perf_counter()
            values = evaluate(reader, traces[name], words, fallback, job['batch_bags'])
            samples[name].append(time.perf_counter() - begin)
    for name, plan in traces.items():
        values = evaluate(reader, plan, words, fallback, job['batch_bags'])
        ref = np.load(common / (name + '-reference.npy'))
        result['query'][name] = {'samples_seconds': samples[name],
            'p50_ms': float(np.median(samples[name]) * 1000),
            'p95_ms': float(np.quantile(samples[name], .95) * 1000),
            'accepted_bags_per_second': float(plan['accepted'].sum() / np.median(samples[name])),
            'original_occurrences': job['occurrences'][name], 'reducer_occurrences': len(plan['ids']),
            'fidelity': fidelity(ref, values, plan['accepted'])}
        dynamic = []
        for _ in range(job['repeat']):
            begin = time.perf_counter()
            rebuilt = schedule(*original_traces[name], partition, fallback=job['fallback'], code_mapping=code_mapping)
            dynamic_values = evaluate(reader, rebuilt, words, fallback, job['batch_bags'])
            dynamic.append(time.perf_counter() - begin)
        np.testing.assert_array_equal(dynamic_values, values)
        result['query'][name]['with_schedule_compilation'] = {
            'samples_seconds': dynamic, 'p50_ms': float(np.median(dynamic) * 1000),
            'p95_ms': float(np.quantile(dynamic, .95) * 1000)}
    # Measured BEFORE full quality decoding/outputs; mmap source weights never opened here.
    result['query_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    result['codec_and_plan_array_bytes'] = reader.codes.nbytes + sum(sum(x.nbytes for x in p.values()) for p in traces.values())
    all_plan = dict(np.load(job['all_plan'], allow_pickle=False))
    hidden = evaluate(reader, all_plan, words, fallback, job['batch_bags'])
    outputs = np.load(common / 'output-rows.npy', mmap_mode='r')
    original = np.load(common / 'all-reference.npy', mmap_mode='r')
    result['known_word_aggregate_fidelity'] = fidelity(original, hidden, all_plan['accepted'])
    result['quality'] = {}
    for name in ('validation', 'test'):
        evaluation = dict(np.load(common / (name + '-evaluation.npz')))
        ref_scores = dict(np.load(common / (name + '-scores.npz')))
        scores = objective(hidden, outputs, evaluation)
        result['quality'][name] = quality_summary(scores, ref_scores)
        result['quality'][name]['pairs'] = len(scores['loss'])
        # Keep paired observations for post-run block intervals, not just means.
        np.savez(Path(job['report']).with_name(job['name'] + '-' + name + '-scores.npz'), **scores)
    write_json(job['report'], result)


def files_size(directory):
    return sum(f.stat().st_size for f in Path(directory).iterdir() if f.is_file())


def run_checkpoint(path, split, cfg, output):
    import torch
    from .benchmark_heldout_compression import evaluation_set
    from .train_ngram_similarity import ngrams
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    rows, outputs = checkpoint['input_rows'].numpy(), checkpoint['output_rows'].numpy()
    words, grams = checkpoint['words'], checkpoint['grams']
    offset, dim = len(words) + 1, rows.shape[1]
    source = np.ascontiguousarray(rows[offset:], dtype=np.float32)
    if not np.isfinite(rows).all() or not np.isfinite(outputs).all():
        raise ValueError('nonfinite trained checkpoint')
    # Preserve exact training multiplicities; protect word rows by not grouping them.
    bags = [[int(i) - offset for i in bag if i >= offset] for bag in checkpoint['bags'].numpy()]
    partition = BagPartition.fit(bags, len(source))
    means = partition.means(source)
    if any(not partition.certifies(bag) for bag in bags):
        raise ValueError('registered training bag is not certified')
    output.mkdir(parents=True, exist_ok=False)
    common = output / 'common'
    common.mkdir()
    np.save(common / 'word-rows.npy', rows[1:offset])
    np.save(common / 'output-rows.npy', outputs)
    (common / 'registry.json').write_text(json.dumps([words, grams], ensure_ascii=True))
    common_bytes = files_size(common)
    partition.save(common / 'domain.npz')
    certificate_bytes = (common / 'domain.npz').stat().st_size
    multi = partition.sizes[partition.mapping] > 1
    np.save(common / 'fallback.npy', source[multi])
    fallback_bytes = (common / 'fallback.npy').stat().st_size
    lookup = {g: i for i, g in enumerate(grams)}
    word_lookup = {w: i for i, w in enumerate(words)}
    known = sorted(set(split['test']) & set(words))
    oov = sorted(set(split['test']) - set(words))
    rng = np.random.default_rng(cfg.seed)
    known = [known[i] for i in sorted(rng.choice(len(known), min(len(known), cfg.query_bags), replace=False))]
    oov = [oov[i] for i in sorted(rng.choice(len(oov), min(len(oov), cfg.query_bags), replace=False))]
    traces = {'known': ([bags[word_lookup[w]] for w in known], [word_lookup[w] for w in known]),
              'oov': ([[lookup[g] for g in ngrams(w, checkpoint['order']) if g in lookup] for w in oov], [-1] * len(oov))}
    write_json(common / 'trace-bags.json', traces)
    all_trace = (bags, list(range(len(words))))
    # Source references use the exact same FP64 weighted reducer as every codec.
    fit_codec(output / 'source-oracle', source, 'fp32', source[:1])
    oracle = FaissRows(output / 'source-oracle')
    word_rows = rows[1:offset]
    for name, (trace_bags, trace_words) in {**traces, 'all': all_trace}.items():
        plan = schedule(trace_bags, trace_words)
        np.save(common / (name + '-reference.npy'), evaluate(oracle, plan, word_rows, batch_bags=cfg.batch_bags))
    counts = Counter(split['train'])
    evaluations = {}
    for name in ('validation', 'test'):
        data = evaluation_set(split[name], words, counts, cfg.eval_pairs, cfg.negatives, cfg.seed + (name == 'test'))
        evaluations[name] = {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}
        np.savez(common / (name + '-evaluation.npz'), pairs=data['pairs'], negative=data['negative'], positions=data['positions'])
        scores = objective(np.load(common / 'all-reference.npy'), outputs, data)
        np.savez(common / (name + '-scores.npz'), **scores)
    train_ids = rng.choice(len(source), min(len(source), cfg.train_rows), replace=False)
    result = {'checkpoint_sha256': digest_file(path), 'order': checkpoint['order'], 'seed': checkpoint['seed'],
              'dim': dim, 'ngram_rows': len(source), 'partition': partition.summary(),
              'common_word_output_registry_bytes': common_bytes,
              'certificate_bytes': certificate_bytes, 'full_domain_fallback_bytes': fallback_bytes,
              'training_original_ids_sha256': hashlib.sha256(train_ids.astype('<u8').tobytes()).hexdigest(),
              'training_rows': len(train_ids), 'evaluation': evaluations,
              'traces': {name: {'bags': len(tb), 'empty_ngram_bags': sum(not b for b in tb),
                   'certified': sum(partition.certifies(b) for b in tb),
                   'bag_word_ids_sha256': hashlib.sha256(json.dumps([tb, tw]).encode()).hexdigest()}
                   for name, (tb, tw) in traces.items()}, 'variants': [], 'failures': []}
    jobs = []
    for grouped in (False, True):
        prefix = 'group' if grouped else 'direct'
        fitted = means if grouped else source
        train = means[partition.mapping[train_ids]] if grouped else source[train_ids]
        for name in cfg.codecs.split(','):
            codec_dir = output / (prefix + '-' + name)
            try:
                metadata = fit_codec(codec_dir, fitted, name, train, seed=cfg.seed,
                                iterations=cfg.iterations, opq_iterations=cfg.opq_iterations)
                # Verify persistence on all stored rows, without retaining a full decode.
                codec = FaissRows(codec_dir)
                h = hashlib.sha256()
                squared, norm = 0., 0.
                for begin, decoded in codec.decoded():
                    if not np.isfinite(decoded).all():
                        raise ValueError('nonfinite codec reconstruction')
                    a = fitted[begin:begin + len(decoded)].astype(np.float64)
                    squared += float(np.square(a - decoded).sum())
                    norm += float(np.square(a).sum())
                    h.update(decoded.astype('<f4').tobytes())
                metadata['decoded_sha256'] = h.hexdigest()
                metadata['decoded_rows_verified'] = len(fitted)
                metadata['stored_row_relative_l2'] = (squared / max(norm, 1e-30)) ** .5
                if name == 'fp32' and h.hexdigest() != hashlib.sha256(fitted.astype('<f4').tobytes()).hexdigest():
                    raise ValueError('FP32 source bytes changed')
                for hybrid in ((False, True) if grouped else (False,)):
                    variant = prefix + '-' + name + ('-hybrid' if hybrid else '')
                    plans = output / (variant + '-plans')
                    plans.mkdir()
                    trace_plans = {}
                    for trace_name, (tb, tw) in {**traces, 'all': all_trace}.items():
                        plan = schedule(tb, tw, partition if grouped else None, fallback=hybrid)
                        file = plans / (trace_name + '.npz')
                        np.savez(file, **plan)
                        if trace_name != 'all':
                            trace_plans[trace_name] = str(file)
                    job = {'name': variant, 'codec_dir': str(codec_dir), 'common': str(common),
                           'fallback': hybrid, 'grouped': grouped,
                           'coverage': 'full-domain-hybrid' if hybrid else 'certified-only' if grouped else 'full-domain',
                           'model_bytes': common_bytes + files_size(codec_dir) + (certificate_bytes if grouped else 0) + (fallback_bytes if hybrid else 0),
                           'fixed_trace_plan_bytes': sum(Path(f).stat().st_size for f in trace_plans.values()),
                           'occurrences': {key: sum(map(len, tb)) for key, (tb, tw) in traces.items()},
                           'trace_plans': trace_plans, 'all_plan': str(plans / 'all.npz'),
                           'repeat': cfg.repeat, 'batch_bags': cfg.batch_bags,
                           'report': str(output / (variant + '-report.json'))}
                    job_path = output / (variant + '-job.json')
                    write_json(job_path, job)
                    jobs.append((job_path, metadata))
                print(json.dumps({'event': 'fitted', 'order': checkpoint['order'], 'seed': checkpoint['seed'],
                                   'name': prefix + '-' + name, 'training_seconds': metadata['training_seconds']}), flush=True)
            except Exception as exc:
                result['failures'].append({'name': prefix + '-' + name, 'error': repr(exc)})
                print(json.dumps({'event': 'codec-failed', **result['failures'][-1]}), flush=True)
            write_json(output / 'report.json', result)
    for job_path, metadata in np.random.default_rng(cfg.seed).permutation(np.asarray(jobs, dtype=object)):
        process = subprocess.run([sys.executable, '-m', 'tools.benchmark_faiss_cotraining', '--worker', str(job_path)],
                                  capture_output=True, text=True)
        (output / (job_path.stem + '.log')).write_text(process.stdout + process.stderr)
        if process.returncode:
            result['failures'].append({'name': job_path.stem, 'error': 'worker exit ' + str(process.returncode),
                                       'log': job_path.stem + '.log'})
        else:
            value = json.loads(Path(json.loads(job_path.read_text())['report']).read_text())
            value['codec'].update(metadata)
            result['variants'].append(value)
            print(json.dumps({'event': 'queried', 'name': value['name'], 'model_bytes': value['model_bytes'],
                              'validation': value['quality']['validation']}), flush=True)
        write_json(output / 'report.json', result)
    # Selection uses validation only; never select among candidates using test.
    result['selection'] = {}
    for budget in cfg.loss_budgets.split(','):
        limit = float(budget)
        result['selection'][budget] = {}
        for family in ('direct', 'group', 'hybrid'):
            candidates = [v for v in result['variants'] if
                (v['name'].startswith('direct-') if family == 'direct' else
                 v['name'].endswith('-hybrid') if family == 'hybrid' else
                 v['name'].startswith('group-') and not v['name'].endswith('-hybrid'))
                and v['quality']['validation']['loss_relative_change'] <= limit
                and v['quality']['validation']['discrimination_change'] >= -cfg.max_discrimination_drop
                and v['known_word_aggregate_fidelity']['global_relative_l2'] <= cfg.max_aggregate_error]
            chosen = min(candidates, key=lambda v: v['model_bytes']) if candidates else None
            result['selection'][budget][family] = None if chosen is None else {'name': chosen['name'],
                'model_bytes': chosen['model_bytes'], 'test': chosen['quality']['test'],
                'test_quality_budget_passed': chosen['quality']['test']['loss_relative_change'] <= limit
                    and chosen['quality']['test']['discrimination_change'] >= -cfg.max_discrimination_drop}
    result['status'] = 'completed' if not result['failures'] else 'completed-with-failures'
    write_json(output / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--orders', default='4,8,12')
    parser.add_argument('--seeds', default='3,7,11')
    parser.add_argument('--codecs', default=','.join(DEFAULT_CODECS))
    parser.add_argument('--train-rows', type=int, default=16384)
    parser.add_argument('--train-threads', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--opq-iterations', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--query-bags', type=int, default=512)
    parser.add_argument('--batch-bags', type=int, default=64)
    parser.add_argument('--eval-pairs', type=int, default=8192)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=15)
    parser.add_argument('--loss-budgets', default='0.0001,0.001,0.005')
    parser.add_argument('--max-discrimination-drop', type=float, default=.005)
    parser.add_argument('--max-aggregate-error', type=float, default=.05)
    cfg = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU only: CUDA_VISIBLE_DEVICES must be empty')
    if cfg.worker:
        worker(cfg.worker)
        return
    if any(v is None for v in (cfg.checkpoint_root, cfg.corpus, cfg.output_dir)):
        parser.error('checkpoint-root, corpus and output-dir are required')
    import faiss
    import torch
    from .benchmark_heldout_compression import split_tokens
    torch.set_num_threads(1)
    if cfg.train_threads < 1:
        parser.error('positive training thread count required')
    faiss.omp_set_num_threads(cfg.train_threads)
    cfg.output_dir = cfg.output_dir.resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_files': {name: digest_file(Path(__file__).with_name(name)) for name in
                         ('benchmark_faiss_cotraining.py', 'faiss_row_codec.py', 'cotraining_groups.py')},
        'command': shlex.join([sys.executable, *sys.argv]), 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'faiss': faiss.__version__,
        'faiss_compile_options': faiss.get_compile_options(), 'os': platform.platform(),
        'cpu': subprocess.check_output(['lscpu', '-J'], text=True) if sys.platform == 'linux' else platform.processor(),
        'corpus_sha256': digest_file(cfg.corpus), 'device': 'cpu', 'query_threads': 1, 'training_threads': cfg.train_threads,
        'limitations': ['Finite public-corpus SGNS checkpoints, no convergence/semantic/PPL/LLM quality claim.',
          'Standalone original-ID decoding; no IVF/HNSW/ANN search or recall comparison.',
          'Word/output FP32 rows and registry, codec/codebook files, domains and complete hybrid fallback are counted.',
          'Fixed-trace schedules counted separately; no native DATrie/block-codec benefit claimed in this adapter.',
          'Both precompiled replay and original-ID schedule compilation plus query timings are reported.',
          'All controls deduplicate repeated IDs; grouped controls additionally collapse certified group occurrences.',
          'Grouped codec training uses the same sampled ORIGINAL IDs, weighting group means by original-row occupancy.',
          'One fresh query process per variant/checkpoint, warm pages and two warmups; shared CPU, no disk-cold or memory-pressure claim.',
          'Query RSS measured before full quality decoding; excludes offline builder resident state.',
          'Full-domain hybrid uses original nonsingleton rows for rejected queries and quantized singleton rows.',
          'Empty OOV bags return zero and are reported; OOV fidelity is not semantic validation.',
          'Fixed hyperparameters; no exhaustive per-codec tuning or universal superiority claim.',
          'No GPU process accessed; monetary/energy cost and temporary-disk peak unmeasured.']}
    write_json(cfg.output_dir / 'protocol.json', protocol)
    split = split_tokens(cfg.corpus.read_text())
    report = {**protocol, 'runs': []}
    for order in map(int, cfg.orders.split(',')):
        for seed in map(int, cfg.seeds.split(',')):
            name = f'order-{order}-seed-{seed}'
            report['runs'].append(run_checkpoint(cfg.checkpoint_root / name / 'reference.pt', split, cfg, cfg.output_dir / name))
            write_json(cfg.output_dir / 'report.json', report)
    report['status'] = 'completed' if all(r['status'] == 'completed' for r in report['runs']) else 'completed-with-failures'
    write_json(cfg.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
