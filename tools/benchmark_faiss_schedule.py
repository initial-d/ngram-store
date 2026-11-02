"""Paired CPU retiming of frozen validation-selected FAISS row codecs.

No refitting or new model selection. All old/new schedules and complete
registered-word outputs are byte-checked before interleaved timing.
"""
import argparse
import gzip
import hashlib
import importlib.util
import importlib.metadata
import inspect
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

from .benchmark_faiss_cotraining import digest_file, evaluate, fidelity, schedule, write_json
from .cotraining_groups import BagPartition
from .faiss_row_codec import FaissRows


def load_reference(path):
    spec = importlib.util.spec_from_file_location('tools._reference_faiss_schedule', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if inspect.getsource(module.evaluate) != inspect.getsource(evaluate):
        raise ValueError('reducer changed; cannot isolate schedule optimization')
    return module


def worker(path):
    import faiss
    faiss.omp_set_num_threads(1)
    job = json.loads(path.read_text())
    original = json.loads(Path(job['original_job']).read_text())
    reference = load_reference(job['reference_module'])
    common = Path(original['common'])
    reader = FaissRows(original['codec_dir'])
    words = np.load(common / 'word-rows.npy', mmap_mode='r')
    fallback = np.load(common / 'fallback.npy', mmap_mode='r') if original['fallback'] else None
    partition = BagPartition.load(common / 'domain.npz') if original['grouped'] else None
    code_mapping = np.load(original['code_mapping'], mmap_mode='r') if original.get('code_mapping') else None
    options = {'fallback': original['fallback'], 'code_mapping': code_mapping}
    traces = json.loads((common / 'trace-bags.json').read_text())
    for name, trace in traces.items():
        if hashlib.sha256(json.dumps(trace).encode()).hexdigest() != job['expected_trace_hashes'][name]:
            raise ValueError('frozen trace changed: ' + name)
    result = {k: job[k] for k in ('order', 'seed', 'name', 'selected_loss_budget',
              'checkpoint_sha256', 'original_job_sha256')}
    result.update(model_bytes=original['model_bytes'], codec_files_sha256={
        p.name: digest_file(p) for p in sorted(Path(original['codec_dir']).iterdir()) if p.is_file()},
        query={}, all_registered_word_outputs_byte_identical=False)

    # Recover ALL registered original-ID bags from the direct (not grouped) plan.
    direct_job = json.loads(Path(job['direct_job']).read_text())
    direct_all = dict(np.load(direct_job['all_plan'], allow_pickle=False))
    bags = [np.repeat(direct_all['ids'][lo:hi], direct_all['weights'][lo:hi].astype(np.int64)).tolist()
            for lo, hi in zip(direct_all['offsets'][:-1], direct_all['offsets'][1:])]
    all_old = dict(np.load(original['all_plan'], allow_pickle=False))
    all_new = schedule(bags, direct_all['words'], partition, **options)
    for key in all_old:
        if all_old[key].tobytes() != all_new[key].tobytes():
            raise ValueError('registered schedule changed: ' + key)
    old_values = evaluate(reader, all_old, words, fallback, original['batch_bags'])
    new_values = evaluate(reader, all_new, words, fallback, original['batch_bags'])
    if old_values.tobytes() != new_values.tobytes():
        raise ValueError('registered aggregate output bytes changed')
    result['all_registered_word_outputs_byte_identical'] = True
    result['registered_words'] = len(bags)
    result['registered_output_sha256'] = hashlib.sha256(old_values.tobytes()).hexdigest()
    rng = np.random.default_rng(42)
    for trace_name, (trace_bags, trace_words) in traces.items():
        old = reference.schedule(trace_bags, trace_words, partition, **options)
        new = schedule(trace_bags, trace_words, partition, **options)
        saved = dict(np.load(original['trace_plans'][trace_name], allow_pickle=False))
        for key in saved:
            if not (saved[key].tobytes() == old[key].tobytes() == new[key].tobytes()):
                raise ValueError('trace schedule changed: ' + key)
        def compile_old():
            return reference.schedule(trace_bags, trace_words, partition, **options)
        def compile_new():
            return schedule(trace_bags, trace_words, partition, **options)
        def replay():
            return evaluate(reader, saved, words, fallback, original['batch_bags'])
        def dynamic_old():
            return evaluate(reader, compile_old(), words, fallback, original['batch_bags'])
        def dynamic_new():
            return evaluate(reader, compile_new(), words, fallback, original['batch_bags'])
        expected = replay()
        if not (expected.tobytes() == dynamic_old().tobytes() == dynamic_new().tobytes()):
            raise ValueError('trace output bytes changed')
        functions = {'compile_old': compile_old, 'compile_new': compile_new,
                     'replay': replay, 'dynamic_old': dynamic_old, 'dynamic_new': dynamic_new}
        samples = {key: [] for key in functions}
        for _ in range(job['warmup']):
            for f in functions.values():
                f()
        for _ in range(job['repeat']):
            for key in rng.permutation(list(functions)):
                start = time.perf_counter()
                functions[key]()
                samples[key].append(time.perf_counter() - start)
        result['query'][trace_name] = {
            'bag_word_ids_sha256': hashlib.sha256(json.dumps(traces[trace_name]).encode()).hexdigest(),
            'schedule_arrays_byte_identical': True, 'output_bytes_identical': True,
            'fidelity': fidelity(np.load(common / (trace_name + '-reference.npy')), expected, saved['accepted']),
            'timing': {key: {'samples_seconds': values, 'p50_ms': float(np.median(values) * 1000),
                            'p95_ms': float(np.quantile(values, .95) * 1000)} for key, values in samples.items()}}
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    write_json(Path(job['output']), result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path)
    p.add_argument('--roots', type=Path, nargs='+')
    p.add_argument('--reference-module', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--loss-budget', default='0.001')
    p.add_argument('--repeat', type=int, default=15)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--worker', type=Path)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU only: empty CUDA_VISIBLE_DEVICES required')
    if cfg.worker:
        worker(cfg.worker)
        return
    if not all([cfg.report, cfg.roots, cfg.reference_module, cfg.output_dir]):
        p.error('report, roots, reference-module and output-dir required')
    if cfg.repeat < 1 or cfg.warmup < 1:
        p.error('positive repeat and warmup required')
    import faiss
    reference = load_reference(cfg.reference_module)
    raw = cfg.report.read_bytes()
    if cfg.report.suffix == '.gz':
        raw = gzip.decompress(raw)
    frozen = json.loads(raw)
    if frozen['status'] != 'completed':
        raise ValueError('completed frozen comparison required')
    cfg.output_dir = cfg.output_dir.resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'command': shlex.join([sys.executable, *sys.argv]), 'settings': {k: str(v) if isinstance(v, Path) else
          [str(x) for x in v] if k == 'roots' else v for k, v in vars(cfg).items()},
        'source_files_sha256': {name: digest_file(Path(__file__).with_name(name)) for name in
          ['benchmark_faiss_schedule.py', 'benchmark_faiss_cotraining.py', 'cotraining_groups.py', 'faiss_row_codec.py']},
        'reference_module_sha256': digest_file(cfg.reference_module),
        'reference_schedule_sha256': hashlib.sha256(inspect.getsource(reference.schedule).encode()).hexdigest(),
        'frozen_report_sha256': hashlib.sha256(raw).hexdigest(),
        'python': sys.version, 'numpy': np.__version__, 'faiss': faiss.__version__,
        'torch': importlib.metadata.version('torch'),
        'os': platform.platform(), 'cpu': subprocess.check_output(['lscpu', '-J'], text=True),
        'query_threads': 1, 'device': 'cpu', 'runs': [], 'failures': [],
        'limitations': ['Frozen prior validation selections; no refitting, reselection or new test-quality claim.',
          'Paired interleaving within each fresh process on shared warm CPU, not exclusive service latency.',
          'All registered outputs and trace arrays byte-identical; rejects never count as successful work.',
          'Peak RSS includes loading/audit and runtime, not an isolated query-memory measurement.']}
    jobs = []
    for run in frozen['runs']:
        choices = run['selection'][cfg.loss_budget]
        run_name = f"order-{run['order']}-seed-{run['seed']}"
        def find(name):
            found = [root / run_name / (name + '-job.json') for root in cfg.roots
                     if (root / run_name / (name + '-job.json')).exists()]
            if len(found) != 1:
                raise ValueError('exactly one frozen job required: ' + name)
            return found[0]
        # A selected dedup codec's plan addresses code IDs, not original IDs.
        # Recover registered bags from the unchanged original-ID FP32 control.
        direct = find('direct-fp32')
        for family, selected in choices.items():
            if selected is None:
                continue
            original_job = find(selected['name'])
            original = json.loads(original_job.read_text())
            if original['model_bytes'] != selected['model_bytes']:
                raise ValueError('selected model cost mismatch')
            job = {'order': run['order'], 'seed': run['seed'], 'name': selected['name'],
                'selected_loss_budget': cfg.loss_budget, 'checkpoint_sha256': run['checkpoint_sha256'],
                'expected_trace_hashes': {k: v['bag_word_ids_sha256'] for k, v in run['traces'].items()},
                'original_job': str(original_job.resolve()), 'original_job_sha256': digest_file(original_job),
                'direct_job': str(direct.resolve()), 'reference_module': str(cfg.reference_module.resolve()),
                'repeat': cfg.repeat, 'warmup': cfg.warmup,
                'output': str(cfg.output_dir / (run_name + '-' + family + '-result.json'))}
            job_path = cfg.output_dir / (run_name + '-' + family + '-job.json')
            write_json(job_path, job)
            jobs.append(job_path)
    for path in np.random.default_rng(42).permutation(np.asarray(jobs, dtype=object)):
        completed = subprocess.run([sys.executable, '-m', 'tools.benchmark_faiss_schedule', '--worker', str(path)],
                                   capture_output=True, text=True)
        path.with_suffix('.log').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            report['failures'].append({'job': path.name, 'exit_code': completed.returncode})
        else:
            report['runs'].append(json.loads(Path(json.loads(path.read_text())['output']).read_text()))
        write_json(cfg.output_dir / 'report.json', report)
        print(json.dumps({'job': path.name, 'returncode': completed.returncode}), flush=True)
    report['status'] = 'completed' if not report['failures'] else 'completed-with-failures'
    write_json(cfg.output_dir / 'report.json', report)
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
