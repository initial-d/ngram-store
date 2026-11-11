"""Durable follow-through from the registered grid to full independent evidence.

CPU builds and official-data audits overlap existing Qwen GPU lanes. GPU
application lanes start only on idle devices and retain all failed attempts.
No external process is terminated. Raw models/corpora remain task artifacts.
"""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .large_application_frontier import CODECS, publish_job, select_sources
from .tree_node_quality import digest, load, serving_files, write


def peer_reserved_gpus(root):
    """A paired Qwen lane owns its device even between child CUDA contexts."""
    reserved = set()
    for path in (root / 'dispatch.json', root / 'native-parallel' / 'dispatch.json',
                 root / 'application-frontier' / 'gpu-recovery' / 'dispatch.json'):
        if not path.exists(): continue
        state = load(path)
        if state.get('status') == 'running' and state.get('controller_pid'):
            try: command = Path(f"/proc/{state['controller_pid']}/cmdline").read_bytes()
            except FileNotFoundError: command = b''
            if str(root).encode() in command: reserved.update(state.get('reserved_gpus', []))
        for item in state.get('running', {}).values():
            if item.get('gpu') is None: continue
            try: command = Path(f"/proc/{item['pid']}/cmdline").read_bytes()
            except FileNotFoundError: continue
            if str(root).encode() in command: reserved.add(int(item['gpu']))
    return reserved


def acquire_gpu_lease(root, gpu):
    """Cooperating task controllers hold a lease across child CUDA gaps."""
    import fcntl
    path = root / 'inventory/gpu-scheduler-locks' / f'gpu{gpu}.lock'
    path.parent.mkdir(parents=True, exist_ok=True); handle = path.open('a')
    try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: handle.close(); return None
    return handle


class AdoptedChild:
    """Observe an existing worker without terminating or duplicating it.

An orphan may be reaped before its exit code can be read. In that case the
immutable final artifact, rather than a fabricated process exit code, decides
whether follow-through may continue. This distinction is recorded in state.
"""
    def __init__(self, item): self.item = item

    def poll(self):
        path = Path(f"/proc/{self.item['pid']}/cmdline")
        try: command = path.read_bytes()
        except FileNotFoundError: command = b''
        expected = b'\0'.join(v.encode() for v in self.item['command'][3:]) + b'\0'
        if command == expected:
            return None
        argv = self.item['command']
        artifact = Path(argv[argv.index('--output') + 1]) if '--output' in argv else None
        if 'build' in argv and artifact: artifact = artifact / 'built.json'
        if 'gpu-lane' in argv and '--model' in argv:
            artifact = Path(argv[argv.index('--model') + 1]) / 'gpu-complete.json'
        valid = artifact and artifact.exists() and load(artifact).get('status') in ('complete', 'frozen')
        self.item['adoption_terminal_evidence'] = {'artifact': str(artifact),
            'artifact_sha256': digest(artifact) if artifact and artifact.exists() else None,
            'original_exit_code_observed': False, 'valid': bool(valid)}
        return 0 if valid else 1


def source_job(entry, root):
    source = Path(entry['training']); report = load(source / 'training.json')
    data = Path(report['settings']['data'])
    files = {str(data / name): {'bytes': (data / name).stat().st_size, 'sha256': digest(data / name)}
        for name in ('registry.npy', 'word-vocabulary.json', 'prepared.json')}
    for name in ('rows.npy', 'classifier.npz', 'training.json', 'canonical-to-bucket.npy'):
        path = source / name
        if path.exists():
            sha = (report['rows_sha256'] if name == 'rows.npy' else digest(path))
            files[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha}
    job = {'name': source.name, 'source': str(source), 'data': str(data), 'kind': 'source',
        'dim': report['dim'], 'rows': report['rows'], 'files': files,
        'complete_model_bytes': sum(v['bytes'] for v in files.values()),
        'source_report_sha256': digest(source / 'training.json'),
        'quality_reference': 'source', 'decoded_sha256': report['rows_sha256'],
        'source_sha256': digest(__file__)}
    path = root / 'source-jobs' / (source.name + '.json')
    if not path.exists(): publish_job(path, job)
    else:
        frozen = load(path)
        for key in ('source', 'data', 'files', 'source_report_sha256'):
            if frozen[key] != job[key]: raise ValueError('immutable source job changed: ' + key)
    return path


def run(args):
    root = args.root; output = root / 'application-frontier'
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'dispatch.json'
    running = {}; completed = {}; failed = {}; started = set()
    previous_failures = {}
    if getattr(args, 'resume_dispatch', None):
        previous = load(args.resume_dispatch)
        completed.update(previous.get('completed', {}))
        previous_failures = {**previous.get('previous_attempt_failures', {}), **previous.get('failed', {})}
        for name, item in previous.get('running', {}).items():
            adopted = dict(item); adopted.update(child=AdoptedChild(adopted), log=open(os.devnull, 'w'))
            if adopted.get('gpu') is not None:
                adopted['gpu_lock'] = acquire_gpu_lease(root, int(adopted['gpu']))
                if adopted['gpu_lock'] is None: raise RuntimeError('cannot adopt a GPU already leased by a peer')
            running[name] = adopted
        started.update(completed); started.update(running)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': ''}
    verified_sources = {}
    def verified_source_job(entry):
        # Hash each immutable registered source once per controller, rather
        # than re-reading tens of GiB every ten-second scheduling tick.
        source = Path(entry['training']); report = source / 'training.json'
        identity = (str(source), digest(report))
        if identity not in verified_sources:
            verified_sources[identity] = source_job(entry, output)
        return verified_sources[identity]
    def launch(name, command, cores, gpu=None):
        lease = acquire_gpu_lease(root, gpu) if gpu is not None else None
        if gpu is not None and lease is None: return False
        log_path = output / 'logs' / (name + '.log'); log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('a')
        argv = ['taskset', '-c', cores, sys.executable, '-m', *command]
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**env, 'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu)})
        running[name] = {'child': child, 'log': log, 'command': argv, 'pid': child.pid,
            'gpu': gpu, 'gpu_lock': lease, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        started.add(name)
    while True:
        for name, item in list(running.items()):
            code = item['child'].poll()
            if code is None: continue
            item['log'].close()
            if item.get('gpu_lock') is not None: item['gpu_lock'].close()
            final = {k: v for k, v in item.items() if k not in ('child', 'log', 'gpu_lock')}
            final['exit_code'] = code
            (completed if code == 0 else failed)[name] = final
            del running[name]
        selection_path = output / 'source-selection.json'
        if not selection_path.exists() and 'select' not in started:
            reports = list((root / 'training-registered').glob('*/training.json'))
            terminal = ('complete', 'incomplete-row-coverage')
            if len(reports) == 144 and all(load(p)['status'] in terminal for p in reports):
                launch('select', ['tools.large_application_frontier', 'select-sources', '--training',
                    str(root / 'training-registered'), '--output', str(selection_path)], '160-167')
        ready_gpu = []
        if selection_path.exists():
            selected = load(selection_path)['selected']
            exact = [e for e in selected if e['family'][2] == 'exact']
            if len(exact) != 12: raise ValueError('complete two-dataset/two-initialization/three-seed exact sources required')
            # Frozen hash controls get full official CPU test audits as well.
            for i, entry in enumerate(selected):
                path = verified_source_job(entry); name = 'source-test-' + path.stem
                result = output / 'source-quality' / (path.stem + '-test.json')
                if not result.exists() and name not in started:
                    launch(name, ['tools.large_application_frontier', 'evaluate', '--job', str(path),
                        '--split', 'test', '--output', str(result)], str(168 + i))
            source_tests_ready = all((output / 'source-quality' / (Path(e['training']).name + '-test.json')).exists()
                and load(output / 'source-quality' / (Path(e['training']).name + '-test.json'))['status'] == 'complete'
                for e in selected)
            if source_tests_ready:
                for i, prepared in enumerate(sorted((root / 'native-layouts').glob('*/prepared.json'))):
                    selection = root / 'native-serving' / prepared.parent.name / 'selection.json'
                    trace = output / 'native-cache' / prepared.parent.name / 'traces.npz'
                    trace_name = 'trace-' + prepared.parent.name; cache_name = 'cache-' + prepared.parent.name
                    if not selection.exists(): continue
                    if not trace.exists() and trace_name not in started:
                        launch(trace_name, ['tools.benchmark_qwen_cache', 'traces', '--prepared', str(prepared),
                            '--upstream', str(args.upstream), '--output', str(trace)], str(160 + i*8))
                    if trace.exists() and trace_name not in running and cache_name not in started and not (trace.parent / 'complete.json').exists():
                        launch(cache_name, ['tools.benchmark_qwen_cache', 'lane', '--prepared', str(prepared),
                            '--selection', str(selection), '--traces', str(trace), '--output', str(trace.parent),
                            '--core-base', str(160 + i*8)], f'{160+i*8}-{167+i*8}')
            for i, entry in enumerate(exact):
                source = Path(entry['training']); model = output / 'models' / source.name
                build_name = 'build-' + source.name; built = model / 'built.json'
                if not built.exists() and build_name not in started:
                    launch(build_name, ['tools.large_application_frontier', 'build', '--training', str(source),
                        '--selection', str(selection_path), '--output', str(model), '--library', str(args.library),
                        '--structural-library', str(args.structural_library), '--threads', '4'], f'{64+i*8}-{67+i*8}')
                jobs = sorted((model / 'jobs').glob('*.json'))
                # Incremental evaluation consumes only immutable completed jobs.
                quality = model / 'quality'
                eval_prefix = 'evaluate-' + source.name + '-'
                candidates = {}
                for path in jobs:
                    job = load(path); reference = job['quality_reference']
                    if path.stem == reference: candidates[reference] = path
                repair = output / 'native-id-recovery' / 'dispatch.json'
                repair_busy = repair.exists() and any(v.get('model') == source.name
                    for v in load(repair).get('running', {}).values())
                if not repair_busy and not any(n.startswith(eval_prefix) for n in running):
                    for split in ('calibration', 'test'):
                        available = sorted(candidates, key=lambda n: (n != 'direct-fp32', n))
                        next_job = next((name for name in available if not (quality / (name + '-' + split + '.json')).exists()
                            and eval_prefix + name + '-' + split not in started), None)
                        if next_job is not None:
                            launch(eval_prefix + next_job + '-' + split,
                                ['tools.large_application_frontier', 'evaluate', '--job', str(candidates[next_job]),
                                 '--split', split, '--output', str(quality / (next_job + '-' + split + '.json'))],
                                f'{68+i*8}-{71+i*8}')
                            break
                frozen = model / 'selection.json'; freeze_name = 'freeze-' + source.name
                if built.exists() and not frozen.exists() and freeze_name not in started:
                    references = {load(p)['quality_reference'] for p in load(built)['jobs']}
                    if all((quality / (name + '-calibration.json')).exists() and
                           load(quality / (name + '-calibration.json'))['status'] == 'complete' for name in references):
                        launch(freeze_name, ['tools.large_application_frontier', 'freeze', '--built', str(built),
                            '--results', str(quality), '--output', str(frozen)], str(216 + i))
                lane_name = 'gpu-' + source.name
                if not args.skip_gpu_lanes and frozen.exists() and lane_name not in started and not (model / 'gpu-complete.json').exists():
                    controls = [verified_source_job(e) for e in selected
                        if e['family'][:2] == entry['family'][:2] and e['seed'] == entry['seed']]
                    ready_gpu.append((lane_name, model, controls))
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
        occupied = {line.split(',')[0].strip() for line in raw.splitlines() if ',' in line}
        peer_reserved = peer_reserved_gpus(root)
        reserved = {r['gpu'] for r in running.values() if r['gpu'] is not None} | peer_reserved
        free = [int(index) for line in devices.splitlines() for index, uuid in [line.split(',')]
            if int(index) not in reserved and uuid.strip() not in occupied]
        for gpu, (name, model, controls) in zip(free, ready_gpu):
            launch(name, ['tools.run_large_application_frontier', 'gpu-lane', '--model', str(model),
                '--controls', *map(str, controls)], f'{232+gpu*2}-{233+gpu*2}', gpu)
        state = {'status': 'running', 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_sha256': digest(__file__), 'selection_ready': selection_path.exists(),
            'running': {n: {k: v for k, v in r.items() if k not in ('child', 'log', 'gpu_lock')} for n, r in running.items()},
            'completed': completed, 'failed': failed, 'external_gpu_clients': raw.splitlines(),
            'peer_dispatch_reserved_gpus': sorted(peer_reserved),
            'previous_attempt_failures': previous_failures,
            'rule': 'No processes are killed; GPU recovery is separate.' if args.skip_gpu_lanes else 'No processes are killed; free-device gates only.'}
        if selection_path.exists() and not running:
            exact = [e for e in load(selection_path)['selected'] if e['family'][2] == 'exact']
            done = all((output / 'models' / Path(e['training']).name / 'gpu-complete.json').exists()
                and load(output / 'models' / Path(e['training']).name / 'gpu-complete.json')['status'] == 'complete' for e in exact)
            done &= len(list((output / 'native-cache').glob('*/complete.json'))) == 6
            if done or failed:
                state['status'] = 'failed' if failed else 'complete'
                write(state_path, state); return
        write(state_path, state); time.sleep(10)


def gpu_lane(args):
    model = args.model; frozen = load(model / 'selection.json'); built = load(model / 'built.json')
    names = {c['name'] for c in frozen['selection'].values() if c is not None}
    # Always retain direct-code controls paired to every selected composition.
    by_name = {load(path)['name']: path for path in built['jobs']}
    for name in list(names):
        job = load(by_name[name]); reference = job['quality_reference']
        if reference in by_name: names.add(reference)
    jobs = [*args.controls, *[Path(by_name[name]) for name in sorted(names)]]
    full_audited = set(); completed = []; failed = []
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    for cohort in range(3):
        rotated = jobs[cohort:] + jobs[:cohort]
        for batch_size in (1, 32):
            for budget in (.5, 1., 2., 4., 8.):
                for path in rotated:
                    job = load(path)
                    modes = ('resident', 'host') if job['kind'] == 'source' else ('host',)
                    for mode in modes:
                        name = f"c{cohort}-b{batch_size}-g{budget:g}-{job['name']}-{mode}"
                        report = model / 'gpu' / (name + '.json')
                        if report.exists() and load(report)['status'] in ('passed', 'allocator-oom'):
                            completed.append(str(report)); continue
                        identity = (str(path), mode)
                        full = batch_size == 32 and identity not in full_audited
                        for attempt in range(3):
                            attempt_path = model / 'gpu' / 'attempts' / (name + f'-{attempt}.json')
                            attempt_path.parent.mkdir(parents=True, exist_ok=True)
                            command = [sys.executable, '-m', 'tools.benchmark_large_application', '--job', str(path),
                                '--output', str(attempt_path), '--mode', mode, '--budget-gib', str(budget),
                                '--batch-size', str(batch_size)]
                            if not full: command.append('--timing-only')
                            with attempt_path.with_suffix('.log').open('w') as log:
                                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                            if attempt_path.exists() and load(attempt_path)['status'] in ('passed', 'allocator-oom'):
                                report.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(attempt_path, report)
                                if attempt_path.with_suffix('.npz').exists(): shutil.copy2(attempt_path.with_suffix('.npz'), report.with_suffix('.npz'))
                                completed.append(str(report))
                                if full and load(report)['status'] == 'passed': full_audited.add(identity)
                                break
                            failed.append({'attempt': str(attempt_path), 'exit_code': code})
                            time.sleep(10)
                        write(model / 'gpu-progress.json', {'status': 'running', 'completed': completed, 'failed_attempts': failed,
                            'source_sha256': digest(__file__), 'cuda_selector': env.get('CUDA_VISIBLE_DEVICES')})
    expected = 3 * 2 * 5 * sum(2 if load(p)['kind'] == 'source' else 1 for p in jobs)
    write(model / 'gpu-complete.json', {'status': 'complete' if len(completed) == expected else 'failed',
        'expected': expected, 'completed': completed, 'failed_attempts': failed,
        'frozen_selection_sha256': digest(model / 'selection.json'), 'source_sha256': digest(__file__),
        'cuda_selector': env.get('CUDA_VISIBLE_DEVICES'), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'pairing': 'All methods in this model lane use this same physical GPU and cores; three rotated cohorts.'})
    if len(completed) != expected:
        raise RuntimeError('GPU lane has missing cells after retained retries; inspect gpu-progress.json')


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='phase', required=True)
    c = sub.add_parser('run')
    for name in ('root', 'library', 'structural-library', 'upstream'): c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--resume-dispatch', type=Path)
    c.add_argument('--skip-gpu-lanes', action='store_true', help='Continue CPU builds/audits while a separate GPU recovery controller completes pairs.')
    c = sub.add_parser('gpu-lane'); c.add_argument('--model', type=Path, required=True)
    c.add_argument('--controls', nargs='+', type=Path, required=True)
    args = p.parse_args()
    {'run': run, 'gpu-lane': gpu_lane}[args.phase](args)


if __name__ == '__main__': main()
