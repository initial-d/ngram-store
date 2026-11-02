"""Bounded concurrent CPU checkpoint jobs with exact completed-run reuse.

Each checkpoint keeps its own fresh codec/query processes and provenance. This
changes orchestration, not fitting, validation selection or the query contract.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .benchmark_faiss_cotraining import digest_file, write_json
from .faiss_row_codec import DEFAULT_CODECS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reuse-root', type=Path)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--train-threads', type=int, default=4)
    p.add_argument('--orders', default='4,8,12')
    p.add_argument('--seeds', default='3,7,11')
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or min(cfg.jobs, cfg.train_threads) < 1:
        p.error('CPU only with empty CUDA_VISIBLE_DEVICES and positive job/thread counts')
    cfg.output_dir = cfg.output_dir.resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    workers = cfg.output_dir / 'workers'
    workers.mkdir()
    source_files = {name: digest_file(Path(__file__).with_name(name)) for name in
                    ('benchmark_faiss_cotraining.py', 'faiss_row_codec.py', 'cotraining_groups.py')}
    corpus_hash = digest_file(cfg.corpus)
    requested = [(order, seed) for order in map(int, cfg.orders.split(','))
                 for seed in map(int, cfg.seeds.split(','))]
    if len(set(requested)) != len(requested):
        p.error('unique order/seed pairs required')
    orchestration = {'command': shlex.join([sys.executable, *sys.argv]),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'driver_sha256': digest_file(Path(__file__)), 'jobs': cfg.jobs,
        'fit_threads_per_job': cfg.train_threads, 'query_threads_per_worker': 1,
        'shared_cpu': True, 'requested_pairs': requested, 'checkpoint_phases': [], 'failures': []}
    write_json(cfg.output_dir / 'orchestration.json', orchestration)
    reuse = json.loads((cfg.reuse_root / 'protocol.json').read_text()) if cfg.reuse_root else None
    if reuse and (reuse['source_files'] != source_files or reuse['corpus_sha256'] != corpus_hash):
        raise ValueError('reuse changes engine source or corpus')
    required_settings = {'codecs': ','.join(DEFAULT_CODECS), 'train_rows': 16384,
        'train_threads': cfg.train_threads, 'iterations': 20, 'opq_iterations': 10, 'seed': 42,
        'query_bags': 512, 'batch_bags': 64, 'eval_pairs': 8192, 'negatives': 5, 'repeat': 15,
        'loss_budgets': '0.0001,0.001,0.005', 'max_discrimination_drop': .005, 'max_aggregate_error': .05}
    if reuse and any(reuse['settings'].get(k) != value for k, value in required_settings.items()):
        raise ValueError('reuse changes predeclared settings')

    def execute(order, seed):
        name = f'order-{order}-seed-{seed}'
        previous = cfg.reuse_root / name / 'report.json' if cfg.reuse_root else None
        if previous and previous.exists():
            entry = json.loads(previous.read_text())
            if entry.get('status') == 'completed':
                if entry['checkpoint_sha256'] != digest_file(cfg.checkpoint_root / name / 'reference.pt'):
                    raise ValueError('reuse checkpoint hash changed')
                return name, entry, previous.parent, reuse, 'reused-complete'
        output = workers / name
        command = [sys.executable, '-m', 'tools.benchmark_faiss_cotraining',
                   '--checkpoint-root', str(cfg.checkpoint_root), '--corpus', str(cfg.corpus),
                   '--output-dir', str(output), '--orders', str(order), '--seeds', str(seed)]
        for key, value in required_settings.items():
            command.extend(['--' + key.replace('_', '-'), str(value)])
        with (workers / (name + '.log')).open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{name}: worker exit {completed.returncode}')
        phase = json.loads((output / 'report.json').read_text())
        if phase['status'] != 'completed' or phase['source_files'] != source_files or phase['corpus_sha256'] != corpus_hash:
            raise ValueError(name + ': incomplete or changed checkpoint phase')
        return name, phase['runs'][0], output / name, phase, 'new-checkpoint-job'

    entries, phases = {}, {}
    with ThreadPoolExecutor(max_workers=cfg.jobs) as executor:
        futures = {executor.submit(execute, order, seed): (order, seed) for order, seed in requested}
        for future in as_completed(futures):
            try:
                name, entry, directory, phase, mode = future.result()
                required = {prefix + codec + suffix for codec in DEFAULT_CODECS
                            for prefix, suffix in (('direct-', ''), ('group-', ''), ('group-', '-hybrid'))}
                if entry['failures'] or {v['name'] for v in entry['variants']} != required:
                    raise ValueError(name + ': required complete codec grid missing')
            except Exception as error:
                orchestration['failures'].append({'pair': futures[future], 'error': repr(error)})
                write_json(cfg.output_dir / 'orchestration.json', orchestration)
                continue
            (cfg.output_dir / name).symlink_to(directory.resolve(), target_is_directory=True)
            entries[futures[future]] = entry
            phases[futures[future]] = phase
            orchestration['checkpoint_phases'].append({'name': name, 'mode': mode,
                'engine_protocol': {k: v for k, v in phase.items() if k != 'runs'},
                'run_sha256': hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()})
            write_json(cfg.output_dir / 'orchestration.json', orchestration)
            print(json.dumps({'completed': name, 'mode': mode, 'variants': len(entry['variants'])}), flush=True)
    prototype = next(iter(phases.values()), {'source_files': source_files,
        'corpus_sha256': corpus_hash, 'settings': required_settings})
    # Do not mutate the original phase's settings while normalizing the merged
    # grid view: phase commands, paths and source provenance must stay exact.
    report = json.loads(json.dumps({k: v for k, v in prototype.items() if k != 'runs'}))
    report.update(source_commit=orchestration['source_commit'], command=orchestration['command'],
                  checkpoint_orchestration=orchestration,
                  runs=[entries[pair] for pair in requested if pair in entries],
                  status='completed-with-failures' if orchestration['failures'] else 'completed')
    report['settings'].update(orders=cfg.orders, seeds=cfg.seeds, output_dir=str(cfg.output_dir))
    write_json(cfg.output_dir / 'report.json', report)
    if orchestration['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
