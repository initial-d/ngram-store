"""Precompute the unchanged LLM CPU grid while local GPU serving continues.

Use the original six physical-core lanes only after the SGNS main grid drains.
The original dispatcher later consumes these fresh outputs without repeating
queries. Reference coordinators wait for this cohort, preventing core overlap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from .parallel_cpu_serving import check_sample, digest, load


def validate(root):
    from .finalize_memory_application import llm_requests
    path = root / 'cpu-phase-overlap-plan.json'
    plan = load(path)
    original = root / 'cpu-serving-parallel-plan.json'
    hardware = load(original)
    if plan['measurement_host'] != socket.gethostname():
        raise ValueError('Wrong CPU overlap host')
    if plan['original_cpu_plan_sha256'] != digest(original):
        raise ValueError('Original CPU affinity plan changed')
    requests = [r for r in llm_requests(root / 'llm-frontier', root / 'independent-controls')
                if r['mode'] == 'codec' and r['batch'] == 1]
    if requests != plan['requests']:
        raise ValueError('The complete original LLM CPU request grid changed')
    return plan, hardware, original


def ready(root):
    path = root / 'retime-long/report.json'
    if not path.exists():
        return False
    report = load(path)
    if report['status'] == 'failed':
        raise RuntimeError('SGNS main grid failed; overlap remains blocked')
    return bool(report['requests']) and len(report['complete']) == len(report['requests'])


def run(root):
    from .tree_node_quality import environment, write
    plan, hardware, original = validate(root)
    folder = root / 'cpu-phase-overlap'
    folder.mkdir(exist_ok=True)
    target = folder / 'report.json'
    if target.exists():
        raise ValueError('Preserve previous overlap controller before restarting')
    report = {'status': 'waiting-for-complete-sgns-main-grid',
              'environment': environment(), 'plan_sha256': digest(root / 'cpu-phase-overlap-plan.json'),
              'phase_source_sha256': digest(__file__), 'requests': plan['requests'],
              'complete': [], 'failures': [], 'workers': 6}
    write(target, report)
    deadline = time.monotonic() + 24 * 3600
    while not ready(root):
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU main-grid wait expired; existing workers preserved')
        time.sleep(5)
    report.update(status='running', started=time.time())
    write(target, report)
    phase = 'retime-llm-cpu'
    outputs = root / phase
    outputs.mkdir(exist_ok=True)
    lock = threading.Lock()
    failed = threading.Event()

    def lane_work(lane):
        cpus = hardware['lanes'][lane]['cpus']
        for index, request in enumerate(plan['requests']):
            if failed.is_set():
                return
            if hardware['model_lanes'][phase][request['model']] != lane:
                continue
            output = outputs / f'{index:05d}.json'
            try:
                if output.exists():
                    raise ValueError('Fresh overlap output required; another scheduler is active')
                command = ['taskset', '--cpu-list', ','.join(map(str, cpus)), sys.executable,
                           '-m', 'tools.parallel_cpu_serving', 'worker', '--phase', phase,
                           '--job', request['job'], '--output', str(output), '--plan', str(original),
                           '--lane', lane, '--cohort', 'parallel-shared-host', '--cache', str(request['cache']),
                           '--requests', '32', '--upstream', plan['upstream']]
                log = folder / f'{index:05d}.{time.time_ns()}.log'
                with log.open('w') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                                                     OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                if result.returncode:
                    raise RuntimeError('LLM CPU overlap worker failed: ' + str(log))
                check_sample(load(output), request, plan['original_cpu_plan_sha256'], lane, cpus,
                             'parallel-shared-host')
                with lock:
                    report['complete'].append({'request': request, 'output': str(output), 'sha256': digest(output)})
                    write(target, report)
                    print({'llm_cpu_complete': len(report['complete']), 'total': len(plan['requests'])}, flush=True)
            except BaseException as error:
                failed.set()
                with lock:
                    report['failures'].append({'request': request, 'error': repr(error)})
                raise

    try:
        lanes = sorted(set(hardware['model_lanes'][phase].values()))
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lane_work, lanes))
        if len(report['complete']) != len(plan['requests']):
            raise ValueError('Complete unchanged LLM CPU cohort required')
        report.update(status='complete', finished=time.time())
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write(target, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root)
    except BaseException as error:
        path = args.root / 'cpu-phase-overlap/report.json'
        if path.exists():
            value = load(path)
            value.update(status='failed', error=repr(error))
            from .tree_node_quality import write
            write(path, value)
        raise


if __name__ == '__main__':
    main()
