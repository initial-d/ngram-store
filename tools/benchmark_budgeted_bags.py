"""Frozen trained-source OOV coverage under predeclared aggregate-L2 budgets.

Read cached inputs only. Workers are CPU-only and publish metrics/hashes, not
corpus text, row data or model weights. No semantic-quality or latency claim.
"""
import argparse
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

from .budgeted_bags import BudgetedBagRows, build_error_bounds
from .certified_store import compact_native
from .cotraining_groups import BagPartition, CertifiedBagRows
from .external_cotraining import file_hash


BUDGETS = [0., .0001, .001, .005, .01, .02, .03, .05, .1, .2, .5, 1.]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def timed(function):
    for _ in range(2):
        function()
    samples = []
    for _ in range(5):
        before = time.perf_counter()
        function()
        samples.append(time.perf_counter() - before)
    return {'seconds': samples, 'p50_seconds': float(np.median(samples))}


def trace_report(native, strict, source, bags):
    nonempty = np.asarray([bool(bag) for bag in bags])
    strict_mask = strict.partition.certifies_many(bags)
    reference = np.asarray([source[bag].astype(np.float64).mean(axis=0) if bag else np.zeros(source.shape[1])
                            for bag in bags])
    result = native.assess(bags, mean=True)
    error = np.linalg.norm(result.vectors.astype(np.float64) - reference, axis=1)
    source_norm = np.linalg.norm(reference, axis=1)
    relative_error = np.full(len(bags), np.inf)
    np.divide(error, source_norm, out=relative_error, where=source_norm > 0)
    relative_error[(error == 0) & (source_norm == 0)] = 0
    if np.any(error > result.absolute_l2_bounds) or np.any(relative_error > result.relative_l2_bounds):
        raise AssertionError('actual mean error exceeds a claimed bound')
    summed = native.assess(bags)
    source_sum = np.asarray([source[bag].astype(np.float64).sum(axis=0) if bag else np.zeros(source.shape[1])
                             for bag in bags])
    sum_error = np.linalg.norm(summed.vectors.astype(np.float64) - source_sum, axis=1)
    if np.any(sum_error > summed.absolute_l2_bounds):
        raise AssertionError('actual sum error exceeds a claimed bound')
    safe_bags = [bag for bag, safe in zip(bags, strict_mask) if safe]
    if result.vectors[strict_mask].tobytes() != strict.aggregate(safe_bags, mean=True).tobytes():
        raise AssertionError('strict accepted native output bytes changed')
    points = []
    for budget in BUDGETS:
        accepted = result.accepts(max_relative_l2=budget) & nonempty
        points.append({'budget': budget, 'accepted_nonempty': int(accepted.sum()),
            'coverage': float(accepted.sum() / max(nonempty.sum(), 1)),
            'new_nonempty_vs_strict': int((accepted & ~strict_mask).sum()),
            'maximum_actual_relative_l2': float(relative_error[accepted].max()) if accepted.any() else None,
            'oracle_eligible_nonempty': int(((relative_error <= budget) & nonempty).sum())})
    return {'queries': len(bags), 'nonempty': int(nonempty.sum()), 'empty': int((~nonempty).sum()),
        'occurrences': sum(map(len, bags)), 'strict_nonempty': int((strict_mask & nonempty).sum()),
        'strict_coverage': float((strict_mask & nonempty).sum() / max(nonempty.sum(), 1)),
        'points': points, 'all_absolute_and_relative_bounds_pass': True,
        'strict_native_output_bytes_identical': True,
        'absolute_errors': error.tolist(),
        'relative_errors': [float(v) if np.isfinite(v) else None for v in relative_error],
        'absolute_l2_bounds': result.absolute_l2_bounds.tolist(),
        'relative_l2_bounds': [float(v) if np.isfinite(v) else None for v in result.relative_l2_bounds],
        'nonempty_mask': nonempty.tolist(), 'strict_mask': strict_mask.tolist(),
        'cost': {'full_assess_all_queries': timed(lambda: native.assess(bags, mean=True)),
            'strict_compile_and_query_accepted_subset': timed(lambda: strict.aggregate(safe_bags, mean=True)),
            'source_fp64_all_queries': timed(lambda: [source[bag].astype(np.float64).mean(axis=0)
                if bag else np.zeros(source.shape[1]) for bag in bags]),
            'scope': 'Different query/contract scopes. Full assess includes rejected candidates; shared warm CPU.'}}


def run_worker(cfg, training):
    import hashlib
    import re
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    convention, order, seed = cfg.worker.split(':')
    order, seed = int(order), int(seed)
    expected = next(r for r in training['runs'] if
                    (r['convention'], r['order'], r['seed']) == (convention, order, seed))
    name = f'order-{order}-seed-{seed}'
    checkpoint_path = cfg.training_root / convention / name / 'reference.pt'
    checkpoint_hash = file_hash(checkpoint_path)
    if checkpoint_hash != expected['checkpoint_sha256']:
        raise ValueError('cached trained checkpoint digest differs from prior evidence')
    corpus = cfg.training_root / 'corpus.txt'
    if file_hash(corpus) != training['tokenized_corpus_sha256']:
        raise ValueError('cached corpus digest differs from prior evidence')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    all_rows = checkpoint['input_rows'].numpy()
    words, grams = checkpoint['words'], checkpoint['grams']
    offset = len(words) + 1
    source = np.ascontiguousarray(all_rows[offset:], dtype='<f4')
    bags = [[int(i) - offset for i in bag if i >= offset] for bag in checkpoint['bags'].numpy()]
    def ngrams(word):
        bounded = '<' + word + '>'
        return [bounded[i:i+n] for i in range(len(bounded)) for n in range(2, order+1) if i+n <= len(bounded)]
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", corpus.read_text().lower())
    test = tokens[int(len(tokens) * .9):]
    known = sorted(set(test) & set(words))
    oov = sorted(set(test) - set(words))
    rng = np.random.default_rng(42)
    known = [known[i] for i in sorted(rng.choice(len(known), min(len(known), 512), replace=False))]
    oov = [oov[i] for i in sorted(rng.choice(len(oov), min(len(oov), 512), replace=False))]
    lookup, word_lookup = {g: i for i, g in enumerate(grams)}, {w: i for i, w in enumerate(words)}
    traces = {'known': [bags[word_lookup[w]] for w in known],
              'oov': [[lookup[g] for g in ngrams(w) if g in lookup] for w in oov]}
    if convention == 'legacy':
        old = json.loads((cfg.previous_traces_root / name / 'common/trace-bags.json').read_text())
        if any(traces[k] != old[k][0] for k in traces):
            raise ValueError('known/OOV queries differ from the prior frozen trace')
    output = cfg.output_dir / convention / name
    output.mkdir(parents=True, exist_ok=False)
    source.tofile(output / 'source.f32')
    np.asarray([i for bag in bags for i in bag], dtype='<u8').tofile(output / 'ids.u64')
    np.asarray([0, *np.cumsum([len(b) for b in bags])], dtype='<u8').tofile(output / 'offsets.u64')
    common = output / 'common'
    common.mkdir()
    all_rows[1:offset].tofile(common / 'word-rows.f32')
    checkpoint['output_rows'].numpy().tofile(common / 'output-rows.f32')
    (common / 'registry.json').write_text(json.dumps([words, grams], ensure_ascii=True))
    common_bytes = sum(p.stat().st_size for p in common.iterdir())
    build = compact_native(output / 'source.f32', source.shape[1], output / 'ids.u64', output / 'offsets.u64', output / 'store')
    bounds = build_error_bounds(output / 'source.f32', output / 'store', output / 'bounds')
    partition = BagPartition.fit(bags, len(source))
    reference_means = partition.means(source)
    with CertifiedBagRows(output / 'store') as strict, BudgetedBagRows(output / 'store', output / 'bounds') as native:
        for begin in range(0, len(source), 1024):
            end = min(len(source), begin+1024)
            if strict.table.rows(range(begin, end)) != reference_means[partition.mapping[begin:end]].tobytes():
                raise AssertionError('native canonical rows differ from legacy projection')
        reports = {kind: trace_report(native, strict, source, trace) for kind, trace in traces.items()}
    total_native = build['complete_serving_bytes'] + bounds['sidecar_bytes']
    raw_complete = source.nbytes + common_bytes
    report = {'status': 'passed', 'convention': convention, 'order': order, 'seed': seed,
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'os': platform.platform(),
        'checkpoint_sha256': checkpoint_hash, 'source_sha256': file_hash(output / 'source.f32'),
        'source_commit': cfg.source_commit, 'partition': partition.summary(),
        'trace_sha256': {kind: hashlib.sha256(json.dumps(trace, separators=(',', ':')).encode()).hexdigest()
                         for kind, trace in traces.items()},
        'prior_legacy_trace_arrays_identical': convention == 'legacy',
        'canonical_rows_identical': True, 'build': build, 'bounds_build': bounds,
        'capacity': {'raw_ngram_fp32_bytes': source.nbytes, 'unchanged_word_output_registry_bytes': common_bytes,
            'native_certified_bytes': build['complete_serving_bytes'], 'sidecar_bytes': bounds['sidecar_bytes'],
            'raw_complete_model_bytes': raw_complete, 'budgeted_complete_model_bytes': total_native + common_bytes,
            'complete_model_saving_vs_raw_fp32': 1 - (total_native + common_bytes) / raw_complete,
            'source_fallback_retained': False}, 'traces': reports,
        'worker_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'rss_scope': 'Fresh checkpoint worker includes Torch, source/checkpoint, maps, build, oracle and assessments; not isolated native-build RSS.'}
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--previous-traces-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    cfg = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU-only run requires CUDA_VISIBLE_DEVICES to be empty')
    training = json.loads(cfg.training_report.read_text())
    if cfg.worker:
        run_worker(cfg, training)
        return
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'category': 'exploratory frozen-public-corpus trained-checkpoint CPU query coverage',
        'source_commit': cfg.source_commit, 'command': shlex.join([sys.executable, *sys.argv]),
        'training_report_sha256': file_hash(cfg.training_report), 'corpus_sha256': training['tokenized_corpus_sha256'],
        'source_files_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
            ('budgeted_bags.py', 'benchmark_budgeted_bags.py', 'cotraining_groups.py', 'certified_store.py')},
        'python': sys.version, 'numpy': np.__version__, 'os': platform.platform(),
        'cpu': subprocess.check_output(['lscpu', '--json'], text=True),
        'threads': 1, 'budgets': BUDGETS, 'runs': [], 'failures': [],
        'limitations': ['Existing finite SGNS weights; no new training, convergence or independent corpus.',
            'N-gram component aggregate error, not semantic/SGNS/LLM quality or cosine similarity.',
            'Known/OOV query n-grams absent from registry are omitted in every reference.',
            'Sidecar has capacity/build/query costs; rejected queries have no retained source fallback.',
            'Fresh worker RSS includes Torch/source/oracle; prior 31.21x construction figure does not apply.',
            'Shared warm CPU, different timing scopes, no exclusive/cold-disk, GPU, money or energy result.',
            'Author-run frozen-checkpoint follow-up; no third-party adoption or universal superiority.']}
    write_json(cfg.output_dir / 'report.json', report)
    started = time.perf_counter()
    for convention in ('legacy', 'average'):
        for order in (4, 8, 12):
            for seed in (3, 7, 11):
                key = f'{convention}:{order}:{seed}'
                command = [sys.executable, '-m', 'tools.benchmark_budgeted_bags',
                    '--training-root', str(cfg.training_root), '--training-report', str(cfg.training_report),
                    '--previous-traces-root', str(cfg.previous_traces_root), '--output-dir', str(cfg.output_dir),
                    '--source-commit', cfg.source_commit, '--worker', key]
                result = subprocess.run(command, capture_output=True, text=True)
                (cfg.output_dir / (key.replace(':', '-') + '.log')).write_text(result.stdout + result.stderr)
                if result.returncode:
                    report['status'] = 'failed'
                    report['failures'].append({'worker': key, 'exit_code': result.returncode,
                        'log': key.replace(':', '-') + '.log'})
                    write_json(cfg.output_dir / 'report.json', report)
                    raise RuntimeError(f'worker {key} failed; preserve the logged failure')
                worker = json.loads((cfg.output_dir / convention / f'order-{order}-seed-{seed}' / 'report.json').read_text())
                report['runs'].append(worker)
                write_json(cfg.output_dir / 'report.json', report)
                print(json.dumps({'completed': key, 'oov_strict': worker['traces']['oov']['strict_coverage'],
                    'oov_1pct': worker['traces']['oov']['points'][4]['coverage'],
                    'oov_5pct': worker['traces']['oov']['points'][7]['coverage']}), flush=True)
    report['status'] = 'passed'
    report['seconds'] = time.perf_counter() - started
    write_json(cfg.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
