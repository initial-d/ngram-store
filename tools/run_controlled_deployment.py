"""Durable task-local dispatcher: concurrent CPU work and one job per free GPU.

Never kills processes. Unrecognized GPU clients are preserved. Commands, exit
codes and incomplete/failed cells remain visible in an atomic state manifest.
"""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

from .tree_node_quality import digest, load, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    args = p.parse_args(); root = args.root
    running = {}; completed = {}; failed = {}; started = set()
    state_path = root / 'dispatch.json'
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    def launch(name, command, cpu, gpu=None):
        logfile = root / 'logs' / (name + '.log')
        logfile.parent.mkdir(parents=True, exist_ok=True)
        log = logfile.open('a')
        child_env = {**env, 'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu)}
        argv = ['taskset', '-c', cpu, *command]
        process = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        running[name] = {'process': process, 'log': log, 'pid': process.pid, 'gpu': gpu,
                         'command': argv, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        started.add(name)
    while True:
        for name, item in list(running.items()):
            code = item['process'].poll()
            if code is None: continue
            item['log'].close()
            final = {k: v for k, v in item.items() if k not in ('process', 'log')}
            final['exit_code'] = code
            (completed if code == 0 else failed)[name] = final
            del running[name]
        for i, path in enumerate(sorted((root / 'native-layouts').glob('*/prepared.json'))):
            name = 'cpu-' + path.parent.name
            if name not in started:
                launch(name, [sys.executable, '-m', 'tools.controlled_native_deployment', 'lane',
                    '--prepared', str(path), '--output', str(root / 'native-serving' / path.parent.name),
                    '--upstream', str(args.upstream), '--cpu-only'], str(i))
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
        uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in gpus.splitlines()}
        used = {line.split(',')[0].strip() for line in raw.splitlines() if ',' in line}
        busy = {v['gpu'] for v in running.values() if v['gpu'] is not None}
        available = [g for g in range(8) if g not in busy and uuids[g] not in used]
        pending = []
        for path in sorted((root / 'native-serving').glob('*/selection.json')):
            name = 'gpu-' + path.parent.name
            if name not in started:
                pending.append((name, [sys.executable, '-m', 'tools.controlled_native_deployment', 'lane',
                    '--prepared', str(root / 'native-layouts' / path.parent.name / 'prepared.json'),
                    '--output', str(path.parent), '--upstream', str(args.upstream)]))
        for dataset in ('ag_news', 'dbpedia_14'):
            if not (root / 'data' / dataset / 'prepared.json').exists(): continue
            for initialization, rate in ((init, rate) for init in ('zero', 'uniform') for rate in (10., 3., 1.)):
                for seed in (3, 7, 11):
                    for method, buckets in (('exact', 1 << 20), ('hash-matched', 1 << 20),
                                            ('hash-budget-20', 1 << 20), ('hash-budget-22', 1 << 22)):
                        suffix = f'{dataset}-{method}-{initialization}-s{seed}-lr{rate:g}'
                        name = 'train-' + suffix; out = root / 'training-registered' / suffix
                        report = out / 'training.json'
                        if report.exists():
                            if load(report).get('status') == 'complete':
                                completed.setdefault(name, {'status': 'externally-completed', 'report_sha256': digest(report)})
                            # Active manual/pre-existing cells are never duplicated.
                            continue
                        if name not in started:
                            pending.append((name, [sys.executable, '-m', 'tools.large_ngram_application', 'train',
                                '--data', str(root / 'data' / dataset), '--output', str(out),
                                '--addressing', method if not method.startswith('hash-budget') else 'hash-budget',
                                '--buckets', str(buckets), '--seed', str(seed), '--lr', str(rate),
                                '--initialization', initialization]))
        for gpu, (name, command) in zip(available, pending):
            launch(name, command, f'{32 + gpu * 4}-{35 + gpu * 4}', gpu)
        state = {'status': 'running', 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_sha256': digest(__file__), 'running': {k: {a: b for a, b in v.items() if a not in ('process', 'log')} for k, v in running.items()},
            'completed': completed, 'failed': failed, 'pending_count': len(pending),
            'external_gpu_clients': raw.splitlines(), 'rule': 'Preserve all external clients; never terminate unknown/training/evaluation processes.'}
        ready = len(list((root / 'data').glob('*/prepared.json'))) == 2 and len(list((root / 'native-layouts').glob('*/prepared.json'))) == 6
        if ready and not running and not pending:
            state['status'] = 'failed' if failed else 'complete'
            write(state_path, state)
            return
        write(state_path, state)
        time.sleep(10)


if __name__ == '__main__':
    main()
