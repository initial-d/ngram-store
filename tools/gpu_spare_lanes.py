"""Use spare devices for complete matched batch cohorts without stopping workers.

The original model lane drains its remaining batch-1 jobs while a spare device
measures every batch-8 method, including matching sources. Earlier samples are
retained; publication uses the entire newly registered batch-8 cohort.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def load(path): return json.loads(Path(path).read_text())


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_requests(plan):
    if plan.get('schema_version') != 1 or not plan['spares']:
        raise ValueError('Registered spare-lane schema required')
    originals = plan['original_devices']; spares = plan['spares']
    if (not set(spares) <= set(originals) or len(set(spares.values())) != len(spares)
            or set(spares.values()) & set(originals.values())):
        raise ValueError('Spare and original GPU assignments must be disjoint')
    cores = set()
    for cpus in [*plan['cpu_affinity'].values(), *plan['spare_cpu_affinity'].values()]:
        if not cpus or sorted(set(cpus)) != cpus or cores.intersection(cpus):
            raise ValueError('Concurrent GPU lanes require disjoint CPU affinity')
        cores.update(cpus)
    if set(plan['spare_cpu_affinity']) != set(spares):
        raise ValueError('Complete spare CPU assignments required')
    expected = [(i, r) for i, r in enumerate(plan['requests']) if r['model'] in spares and r['batch'] == 8]
    for model in spares:
        selected = [r for _, r in expected if r['model'] == model]
        if not selected or not any(r['name'] == 'source-gpu' for r in selected):
            raise ValueError('Complete matched source cohort required')
    return expected


def effective_cohort(plan, original, spare):
    """Choose a whole registered batch cohort, without looking at latency."""
    expected = cohort_requests(plan)
    if (original['status'] != 'complete' or spare['status'] != 'complete'
            or original['requests'] != plan['requests']
            or spare['requests'] != [r for _, r in expected]):
        raise ValueError('Complete unchanged original and spare schedules required')
    def indexed(items, requests):
        result = {}
        for item in items:
            key = json.dumps(item['request'], sort_keys=True)
            if key in result: raise ValueError('Duplicate GPU sample')
            result[key] = item
        if set(result) != {json.dumps(r, sort_keys=True) for r in requests}:
            raise ValueError('Incomplete GPU cohort')
        return result
    prior = indexed(original['complete'], plan['requests'])
    replacement = indexed(spare['complete'], [r for _, r in expected])
    complete = []
    for request in plan['requests']:
        key = json.dumps(request, sort_keys=True)
        item = dict(replacement.get(key, prior[key]))
        item['gpu'] = (plan['spares'][request['model']] if key in replacement
                       else plan['original_devices'][request['model']])
        complete.append(item)
    return {**original, 'complete': complete, 'workers': len(plan['original_devices']) + len(plan['spares']),
            'cohort_contract': 'Complete registered model/batch cohorts; earlier original-card batch-8 diagnostics retained',
            'spare_device_assignment': plan['spares'], 'spare_cpu_affinity': plan['spare_cpu_affinity']}


def command(request, output, upstream, *, requests=32, repeat=3, warmup=1):
    result = [sys.executable, '-m', 'tools.benchmark_llm_codec_serving', 'gpu',
              '--job', request['job'], '--output', str(output), '--upstream', str(upstream),
              '--mode', request['mode'], '--cache-bytes', str(request['cache']),
              '--batch-size', str(request['batch']), '--decode-steps', '16',
              '--requests', str(requests), '--repeat', str(repeat), '--warmup', str(warmup),
              '--skip-spare-routing']
    if request['mode'] == 'independent': result += ['--control-run', request['control_run']]
    return result


def check_result(value, request, gpu, cpus):
    if (value['status'] != 'passed' or value['job_sha256'] != digest(request['job'])
            or value['selected_gpu_selector'] != str(gpu) or value['cpu_affinity'] != cpus
            or value['batch_size'] != request['batch'] or value['decode_steps'] != 16
            or value['mode'] != request['mode'] or value['cache_bytes'] != request['cache']):
        raise ValueError('Spare-device worker changed its matched request or hardware')
    if value['gpu_before']['compute_pids'] or any(v['external_compute_pids'] for v in value['gpu_samples']):
        raise ValueError('Spare-device measurement was not isolated')


def route_pending(args):
    """Called only before CUDA initialization in a future original batch-8 job."""
    plan_path = args.output.parent.parent / 'gpu-spare-lane-plan.json'
    if not plan_path.exists(): return False
    plan = load(plan_path)
    cohort_requests(plan)
    if str(args.output.parent) != plan['canonical_directory'] or args.batch_size != 8: return False
    index = int(args.output.stem); request = plan['requests'][index]
    if request['model'] not in plan['spares']: return False
    if str(args.job) != request['job'] or args.mode != request['mode'] or args.cache_bytes != request['cache']:
        raise ValueError('Original request differs from frozen spare-lane plan')
    if (args.requests, args.repeat, args.warmup, args.decode_steps) != (32, 3, 1, 16):
        raise ValueError('Frozen GPU request budgets changed')
    if request['mode'] == 'independent' and str(args.control_run) != request['control_run']:
        raise ValueError('Independent control identity changed')
    from .tree_node_quality import write
    root = plan_path.parent; assigned = plan['spares'][request['model']]
    expected_cpus = plan['cpu_affinity'][request['model']]
    if sorted(os.sched_getaffinity(0)) != expected_cpus:
        raise ValueError('Original model core assignment changed')
    folder = root / 'gpu-spare-lanes'; folder.mkdir(exist_ok=True)
    drain = folder / (request['model'] + '-batch1-drain.json')
    with (folder / (request['model'] + '.lock')).open('w') as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        report = {'status': 'running', 'model': request['model'], 'complete': [], 'plan_sha256': digest(plan_path)}
        write(drain, report)
        for i, other in enumerate(plan['requests']):
            if other['model'] != request['model'] or other['batch'] != 1: continue
            output = Path(plan['canonical_directory']) / f'{i:05d}.json'
            if not output.exists():
                log = folder / f'{request["model"]}-batch1-{i:05d}.{time.time_ns()}.log'
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['original_devices'][request['model']]))
                with log.open('w') as stream:
                    result = subprocess.run(command(other, output, plan['upstream']), env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError('Original-device batch-1 drain failed: ' + str(log))
            check_result(load(output), other, plan['original_devices'][request['model']], expected_cpus)
            report['complete'].append({'request': other, 'output': str(output), 'sha256': digest(output)})
            write(drain, report)
        report['status'] = 'complete'; write(drain, report)
    target = folder / 'outputs' / f'{index:05d}.json'; deadline = time.monotonic() + 24 * 3600
    while not target.exists():
        controller = folder / 'report.json'
        if controller.exists() and load(controller)['status'] == 'failed':
            raise RuntimeError('Spare device failed; preserved diagnostics require repair')
        if time.monotonic() > deadline: raise TimeoutError('Spare-device wait expired; active workers retained')
        time.sleep(5)
    check_result(load(target), request, assigned, plan['spare_cpu_affinity'][request['model']])
    # The already-running original controller checks its child's CPU affinity.
    # Give it an explicit routing receipt, never relabel a GPU measurement.
    receipt = {'status': 'passed', 'measurement_purpose': 'routing-receipt-only',
               'job_sha256': digest(request['job']), 'cpu_affinity': expected_cpus,
               'request': request, 'gpu_spare_plan_sha256': digest(plan_path),
               'spare_measurement_output': str(target), 'spare_measurement_sha256': digest(target),
               'measurement_cpu_affinity': plan['spare_cpu_affinity'][request['model']],
               'measurement_gpu': assigned}
    if args.output.exists():
        if load(args.output) != receipt:
            raise ValueError('An existing original sample must never be overwritten')
    else:
        write(args.output, receipt)
    return True


def run(plan_path):
    from .tree_node_quality import environment, write
    plan = load(plan_path); root = plan_path.parent; output = root / 'gpu-spare-lanes'
    (output / 'outputs').mkdir(parents=True, exist_ok=True)
    expected = cohort_requests(plan)
    report = {'status': 'running', 'environment': environment(), 'started': time.time(),
              'plan_sha256': digest(plan_path), 'phase_source_sha256': digest(__file__),
              'requests': [r for _, r in expected], 'complete': [], 'preflights': [], 'failures': [],
              'workers': len(plan['spares']), 'workers_per_gpu': 1}
    controller = output / 'report.json'; lock = threading.Lock(); write(controller, report)
    def measure(request, path, gpu, cpus, *, preflight=False):
        if not path.exists():
            log = path.with_suffix('.' + str(time.time_ns()) + '.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            cmd = ['taskset', '--cpu-list', ','.join(map(str, cpus)), *command(request, path, plan['upstream'],
                requests=2 if preflight else 32, repeat=1 if preflight else 3, warmup=0 if preflight else 1)]
            with log.open('w') as stream:
                result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('Spare GPU failed: ' + str(log))
        value = load(path); check_result(value, request, gpu, cpus)
        return value
    def lane(model):
        gpu = plan['spares'][model]; cpus = plan['spare_cpu_affinity'][model]
        source = next(r for _, r in expected if r['model'] == model and r['name'] == 'source-gpu')
        native = next(r for _, r in expected if r['model'] == model and r['name'] == 'native-none-fp32' and r['cache'] == 0)
        directory = output / (model + '-preflight'); directory.mkdir(exist_ok=True)
        values = [measure(r, directory / (name + '.json'), gpu, cpus, preflight=True)
                  for r, name in ((source, 'source'), (native, 'native'))]
        for a, b in zip(values[0]['requests'], values[1]['requests']):
            for field in ('input_ids_sha256', 'cached_logit_hashes', 'generated_ids_sha256'):
                if a[field] != b[field]: raise ValueError('New physical GPU source/native preflight differs')
        with lock:
            report['preflights'].append({'model': model, 'status': 'passed', 'gpu': gpu,
                'outputs': [{'output': str(directory / (name + '.json')), 'sha256': digest(directory / (name + '.json'))} for name in ('source', 'native')]})
            write(controller, report)
        for index, request in expected:
            if request['model'] != model: continue
            path = output / 'outputs' / f'{index:05d}.json'; measure(request, path, gpu, cpus)
            with lock:
                report['complete'].append({'request': request, 'output': str(path), 'sha256': digest(path), 'gpu': gpu})
                write(controller, report)
                print({'spare_gpu': gpu, 'complete': len(report['complete']), 'expected': len(expected)}, flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(plan['spares'])) as pool:
            list(pool.map(lane, sorted(plan['spares'])))
        if len(report['complete']) != len(expected): raise ValueError('Spare-device cohort incomplete')
        report['status'] = 'complete'; report['finished'] = time.time()
    except BaseException as error:
        report['status'] = 'failed'; report['failures'].append({'error': repr(error), 'time': time.time()})
        raise
    finally: write(controller, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); run(args.plan)


if __name__ == '__main__': main()
