"""Resume failed final serving with unchanged quality gates and retained evidence."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .finish_memory_application import load, sha
from .tree_node_quality import write


def validate_ready(root):
    ready = load(root / 'quality-ready.json')
    if ready['status'] != 'quality-complete': raise ValueError('Complete original quality gate required')
    for name, checksum in ready['gates'].items():
        path = root / name
        if sha(path) != checksum or load(path)['status'] != 'complete':
            raise ValueError('Original quality gate changed: ' + name)
    for family, checksum in ready['frozen_selections'].items():
        if sha(root / family / 'selection.json') != checksum: raise ValueError('Frozen choice changed')
    return ready


def run(args):
    root = args.root; validate_ready(root)
    old = load(root / 'final-serving-dispatch.json')
    if old['status'] != 'failed': raise ValueError('Only a failed, inactive final dispatch may be resumed')
    backup = root / 'recovery-20261110'; backup.mkdir(exist_ok=True); stamp = str(time.time_ns())
    snapshots = {}
    for name in ('final-serving-dispatch.json', 'retime-llm-gpu/report.json', 'gpu-spare-lanes/report.json',
                 'publication-prepared/report.json'):
        path = root / name
        if path.exists():
            target = backup / (name.replace('/', '-') + '-' + stamp + '.json')
            target.write_bytes(path.read_bytes()); snapshots[name] = {'path': str(target), 'sha256': sha(target)}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    write(root / 'gpu-recovery.json', {'source_commit': commit, 'source_sha256': sha(Path(__file__)),
        'registered': time.time(), 'snapshots': snapshots,
        'reason': 'Externally competing GPU programs interrupted measurements; passed outputs are retained and interrupted attempts are archived'})
    report = {'status': 'resuming-final-serving', 'started': time.time(), 'source_commit': commit,
        'complete': [v for v in old['complete'] if v['name'] not in ('retime-llm-gpu', 'retime-llm-cpu', 'retime-long')],
        'failures': [], 'previous_failures': old.get('previous_failures', []) + old.get('failures', [])}
    write(root / 'final-serving-dispatch.json', report)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    def phase(name, cmd):
        log = root / (name + '-resume-' + stamp + '.log')
        with log.open('w') as stream: result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        item = {'name': name, 'command': cmd, 'returncode': result.returncode, 'log': str(log), 'finished': time.time()}
        if result.returncode:
            report['failures'].append(item); raise RuntimeError('Recovery phase failed: ' + str(log))
        report['complete'].append(item); write(root / 'final-serving-dispatch.json', report)
    common = ['--controls', str(root / 'independent-controls'), '--ready', str(root / 'quality-ready.json'),
              '--upstream', str(args.upstream), '--requests', '32']
    spare = None
    try:
        with (root / ('spare-gpu-resume-' + stamp + '.log')).open('w') as stream:
            spare = subprocess.Popen([sys.executable, '-m', 'tools.gpu_spare_lanes', '--plan', str(root / 'gpu-spare-lane-plan.json')],
                env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        phase('retime-llm-gpu', [sys.executable, '-m', 'tools.finalize_memory_application', 'retime-llm-gpu',
            '--frontier', str(root / 'llm-frontier'), '--output', str(root / 'retime-llm-gpu'),
            '--gpu-addendum', str(Path.cwd() / 'benchmarks/memory_application_serving_addendum_v1.json'),
            '--gpu-affinity', str(root / 'gpu-worker-affinity.json'), *common])
        if spare.wait() != 0 or load(root / 'gpu-spare-lanes/report.json')['status'] != 'complete':
            raise RuntimeError('Complete spare cohort required before CPU timing')
        (root / 'all-compute-finished').write_text('All registered GPU measurements completed.\n')
        for name, family in (('retime-llm-cpu', 'llm-frontier'), ('retime-long', 'long-frontier')):
            report['status'] = 'running-' + name; write(root / 'final-serving-dispatch.json', report)
            phase(name, [sys.executable, '-m', 'tools.finalize_memory_application', name, '--frontier', str(root / family),
                '--output', str(root / name), *common])
        report.update(status='complete', finished=time.time()); write(root / 'final-serving-dispatch.json', report)
    except BaseException as error:
        report.update(status='failed', error=repr(error), finished=time.time()); write(root / 'final-serving-dispatch.json', report)
        raise
    output = root / 'publication-prepared'
    if output.exists(): output.rename(root / ('publication-prepared-before-resume-' + stamp))
    with (root / ('publication-resume-' + stamp + '.log')).open('w') as stream:
        subprocess.run([sys.executable, '-m', 'tools.finish_memory_application', '--root', str(root),
            '--sources', str(args.sources), '--output', str(output), '--path-map', str(root / 'publication-path-map.json'),
            '--release-url', args.release_url], cwd=args.publication_source, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'sources', 'upstream', 'publication-source'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-url', required=True); args = parser.parse_args()
    with (args.root / 'final-serving-recovery.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); run(args)


if __name__ == '__main__': main()
