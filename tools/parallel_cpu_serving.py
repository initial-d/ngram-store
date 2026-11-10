"""Pinned fresh CPU workers, with frozen-choice serial reference measurements.

Parallel samples describe a shared host. Serial references independently audit
the fixed primary-budget comparisons on the same physical cores. Neither phase
changes candidates, quality choices, request traces or repetition counts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import socket


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_requests(requests, selection, phase):
    """Choose references by registered quality budgets, never timing results."""
    names = {}
    for model in selection['models']:
        chosen = {c['name'] for c in model['choices'] if c['name'] and (
            c.get('budget_pp') == .5 if phase == 'retime-llm-cpu'
            else c.get('budget') == .001 and c['cached'] == 0)}
        chosen.update(('raw-fp32', 'raw-fp16', 'raw-int8', 'dense-fp32',
                       'native-none-fp32', 'native-none-fp16', 'native-none-int8'))
        names[model['name']] = chosen
    return [r for r in requests if r['name'] in names[r['model']] and r['cache'] == 0 and (
        phase == 'retime-llm-cpu' or r['cached'] == 0 and (
            r['index'] == 'datrie' or r['name'] == 'raw-fp32' or r['name'].startswith('tree-datrie-')))]


def validate_plan(plan, phase, models, allowed):
    lanes = plan['lanes']; assignment = plan['model_lanes'][phase]
    if plan['schema_version'] != 1 or len(lanes) != plan['workers'] or set(assignment) != set(models):
        raise ValueError('Complete fixed CPU lane plan required')
    used_cpus = set(); used_cores = set()
    for lane, item in lanes.items():
        cpus = item['cpus']; cores = {tuple(c) for c in item['physical_cores']}
        if (not cpus or not set(cpus) <= set(allowed) or used_cpus.intersection(cpus)
                or not cores or used_cores.intersection(cores)):
            raise ValueError('Disjoint allowed logical AND physical CPU cores required')
        used_cpus.update(cpus); used_cores.update(cores)
    if any(lane not in lanes for lane in assignment.values()):
        raise ValueError('Model assigned to unknown CPU lane')


def check_sample(value, request, plan_hash, lane, cpus, cohort):
    if (value['status'] not in ('complete', 'passed') or value['job_sha256'] != digest(request['job'])
            or value['cpu_parallel_plan_sha256'] != plan_hash or value['cpu_lane'] != lane
            or value['cpu_affinity'] != cpus or value['measurement_cohort'] != cohort):
        raise ValueError('CPU worker provenance or isolation changed')


def compare_outputs(parallel, reference):
    if parallel['decoded_row_sha256'] != reference['decoded_row_sha256']:
        raise ValueError('Serial reference changed decoded original row bytes')
    ratios = {}
    for kind, value in parallel['query'].items():
        actual = reference['query'][kind]
        if 'output_sha256' in value:
            if value['output_sha256'] != actual['output_sha256']:
                raise ValueError('Serial reference changed complete query output')
            a, b = value['complete_p50_ms'], actual['complete_p50_ms']
        else:
            if value['input_hashes'] != actual['input_hashes'] or value['output_hashes'] != actual['output_hashes']:
                raise ValueError('Serial reference changed complete original-key requests')
            a, b = value['p50_ms'], actual['p50_ms']
        if a <= 0 or b <= 0:
            raise ValueError('Positive timing samples required')
        ratios[kind] = a / b
    return ratios


def run_queue(args, requests, ready, plan_path):
    from .tree_node_quality import environment, write
    plan = load(plan_path); plan_hash = digest(plan_path)
    models = sorted({r['model'] for r in requests})
    if plan.get('measurement_host') and socket.gethostname()!=plan['measurement_host']:
        controller=args.output/'report.json';deadline=time.monotonic()+24*3600
        while time.monotonic()<deadline:
            if controller.exists():
                value=load(controller)
                if value['status']=='failed':raise RuntimeError('Remote CPU evidence failed; diagnostics retained')
                if value['status']=='complete':
                    if value['requests']!=requests or value['cpu_parallel_plan_sha256']!=plan_hash:
                        raise ValueError('Remote CPU schedule or plan changed')
                    return
            time.sleep(10)
        raise TimeoutError('Remote CPU evidence wait expired; workers retained')
    validate_plan(plan, args.phase, models, os.sched_getaffinity(0))
    if not plan.get('parallel_with_gpu_shards') and not (args.ready.parent / 'all-compute-finished').exists():
        raise ValueError('CPU parallel timing starts only after all GPU timing finished')
    selection = args.frontier / 'selection.json'; selection_hash = digest(selection)
    refs = reference_requests(requests, load(selection), args.phase)
    args.output.mkdir(parents=True, exist_ok=True); controller = args.output / 'report.json'
    prior = load(controller) if controller.exists() else {}
    if prior and (prior['requests'] != requests or prior['cpu_parallel_plan_sha256'] != plan_hash
                  or prior['frozen_selection_sha256'] != selection_hash):
        raise ValueError('Existing CPU schedule differs')
    report = {'status': 'running', 'phase': args.phase, 'environment': environment(),
              'requests': requests, 'complete': [], 'failures': [],
              'previous_failures': prior.get('previous_failures', []) + prior.get('failures', []),
              'workers': plan['workers'], 'workers_per_physical_core': 1,
              'cpu_parallel_plan_sha256': plan_hash, 'cpu_parallel_plan': str(plan_path),
              'model_lanes': plan['model_lanes'][args.phase], 'lanes': plan['lanes'],
              'phase_source_sha256': digest(__file__), 'all_quality_ready_sha256': digest(args.ready),
              'frozen_selection_sha256': selection_hash, 'serial_reference_workers': 1,
              'reference_requests': refs, 'serial_references': [],
              'execution_schedule_sha256': hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest(),
              'measurement_contract': 'Pinned per-model lanes on a shared host; primary-budget serial references follow the full grid'}
    if plan.get('measurement_host'):
        report.update(measurement_host=socket.gethostname(),parallel_with_gpu_shards=plan.get('parallel_with_gpu_shards',False))
    retained = {e['output']: e['sha256'] for e in prior.get('complete', []) + prior.get('serial_references', [])}
    lock = threading.Lock(); failures = []
    def measure(index, request, cohort):
        lane = report['model_lanes'][request['model']]; cpus = plan['lanes'][lane]['cpus']
        folder = args.output if cohort == 'parallel-shared-host' else args.output / 'serial-reference'
        folder.mkdir(exist_ok=True); output = folder / f'{index:05d}.json'
        if output.exists():
            if str(output) in retained and digest(output) != retained[str(output)]:
                raise ValueError('Retained CPU evidence changed')
            value = load(output)
            check_sample(value, request, plan_hash, lane, cpus, cohort)
        else:
            command = ['taskset', '--cpu-list', ','.join(map(str, cpus)), sys.executable,
                       '-m', 'tools.parallel_cpu_serving', 'worker', '--phase', args.phase,
                       '--job', request['job'], '--output', str(output), '--plan', str(plan_path),
                       '--lane', lane, '--cohort', cohort, '--cache', str(request['cache']),
                       '--requests', str(args.requests)]
            if args.phase == 'retime-long':
                command += ['--cached', str(request['cached']), '--index', request['index']]
            else:
                command += ['--upstream', str(args.upstream)]
            log = folder / f'{index:05d}.{time.time_ns()}.log'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with log.open('w') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('CPU measurement failed; retained log: ' + str(log))
            value = load(output); check_sample(value, request, plan_hash, lane, cpus, cohort)
        return {'request': request, 'output': str(output), 'sha256': digest(output), 'cpu_lane': lane}
    write(controller, report)
    def lane_work(lane):
        for index, request in enumerate(requests):
            if report['model_lanes'][request['model']] != lane:
                continue
            try:
                result = measure(index, request, 'parallel-shared-host')
                with lock:
                    report['complete'].append(result); write(controller, report)
                    if len(report['complete']) % 20 == 0:
                        print({'phase': args.phase, 'complete': len(report['complete']), 'expected': len(requests)}, flush=True)
            except Exception as error:
                with lock:
                    item = {'request': request, 'error': repr(error), 'time': time.time()}
                    report['failures'].append(item); failures.append(item); write(controller, report)
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        list(pool.map(lane_work, sorted(plan['lanes'])))
    if failures or len(report['complete']) != len(requests):
        report['status'] = 'failed'; write(controller, report)
        raise RuntimeError('Full CPU parallel grid incomplete')
    report['status'] = 'running-serial-references'; write(controller, report)
    if plan.get('local_gpu_finished_marker'):
        marker=args.ready.parent/plan['local_gpu_finished_marker'];deadline=time.monotonic()+24*3600
        while not marker.exists():
            gpu=args.ready.parent/'gpu-prompt-shards/report.json'
            if gpu.exists() and load(gpu)['status']=='failed':raise RuntimeError('GPU shards failed before CPU serial references')
            if time.monotonic()>deadline:raise TimeoutError('Local GPU completion wait expired')
            time.sleep(10)
    originals = {json.dumps(v['request'], sort_keys=True): v for v in report['complete']}
    try:
        for index, request in enumerate(refs):
            result = measure(index, request, 'serial-reference')
            original = originals[json.dumps(request, sort_keys=True)]
            result['parallel_output_sha256'] = original['sha256']
            result['parallel_to_serial_p50_ratios'] = compare_outputs(load(original['output']), load(result['output']))
            report['serial_references'].append(result); write(controller, report)
            if index % 20 == 0:
                print({'phase': args.phase, 'serial_references': index + 1, 'expected': len(refs)}, flush=True)
        if digest(selection) != selection_hash or digest(plan_path) != plan_hash:
            raise ValueError('Frozen selection or CPU plan changed')
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'failed'; report['failures'].append({'error': repr(error), 'stage': 'serial-reference'})
        raise
    finally:
        write(controller, report)


def worker(args):
    from .tree_node_quality import query_job, write
    plan=load(args.plan);expected = plan['lanes'][args.lane]['cpus']
    if sorted(os.sched_getaffinity(0)) != expected:
        raise ValueError('CPU affinity differs from fixed lane')
    before = list(os.getloadavg())
    started=time.time();gate={}
    if args.cohort=='serial-reference' and plan.get('local_gpu_finished_marker'):
        marker=args.plan.parent/plan['local_gpu_finished_marker'];value=load(marker)
        if value['status']!='complete' or value['measurement_host']!=socket.gethostname():raise ValueError('Wrong local GPU gate')
        gate={'local_gpu_gate_sha256':digest(marker)}
    if args.phase == 'retime-long':
        query_job(args.job, args.cached, index_kind=args.index, cache_bytes=args.cache, selected=True)
        job = load(args.job)
        path = Path(job['report']).with_name(job['name'] + f'-cache{args.cached}-{args.index}-rowcache{args.cache}-retiming.json')
        value = load(path)
    else:
        from .benchmark_llm_codec_serving import cpu_worker
        args.cache_bytes = args.cache; args.repeat = 3; args.warmup = 1
        cpu_worker(args); value = load(args.output)
    value.update(cpu_parallel_plan_sha256=digest(args.plan), cpu_lane=args.lane,
                 cpu_affinity=sorted(os.sched_getaffinity(0)), measurement_cohort=args.cohort,
                 worker_source_sha256=digest(__file__), host_load_before=before,
                 host_load_after=list(os.getloadavg()))
    if load(args.plan).get('measurement_host'):
        value.update(measurement_host=socket.gethostname(),sample_started=started,sample_finished=time.time(),**gate)
    write(args.output, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('worker',))
    parser.add_argument('--phase', choices=('retime-long', 'retime-llm-cpu'), required=True)
    for name in ('job', 'output', 'plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--lane', required=True)
    parser.add_argument('--cohort', choices=('parallel-shared-host', 'serial-reference'), required=True)
    parser.add_argument('--cache', type=int, default=0)
    parser.add_argument('--cached', type=int, choices=(0, 1), default=0)
    parser.add_argument('--index', choices=('datrie', 'hash'), default='datrie')
    parser.add_argument('--requests', type=int, default=32)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Empty CUDA_VISIBLE_DEVICES required')
    # A reference coordinator can finish between the dispatcher's existence
    # check and this CLI starting. Consume its verified output before enforcing
    # the fresh-output requirement that applies to numerical workers.
    from .cpu_parallel_references import wait_for_reference
    if wait_for_reference(args):return
    if args.output.exists():
        parser.error('Fresh output and empty CUDA_VISIBLE_DEVICES required')
    from .cpu_fork_serving import submit
    if submit(args):return
    worker(args)


if __name__ == '__main__':
    main()
