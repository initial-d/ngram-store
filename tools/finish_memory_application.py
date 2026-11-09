"""Finish an already dispatched study: verify, analyze, plot and package it.

No model worker or other process is stopped. Numerical publication starts only
after the unchanged complete final-serving controller passes. Release upload
and source merge stay outside this preparation command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def load(path):
    return json.loads(path.read_text())


def sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def wait_for_serving(root, timeout, poll, notify):
    deadline = time.monotonic() + timeout
    controller = root / 'final-serving-dispatch.json'
    previous = None
    while True:
        value = load(controller) if controller.exists() else {'status': 'not-started'}
        state = value['status']
        if state == 'failed':
            raise RuntimeError('Final serving failed; inspect the retained controller: ' + str(controller))
        if state == 'complete':
            if value.get('failures'):
                raise ValueError('Complete controller still declares unresolved failures')
            phases = [item['name'] for item in value['complete']]
            for phase in ('retime-llm-gpu', 'retime-llm-cpu', 'retime-long'):
                if phases.count(phase) != 1:
                    raise ValueError('Missing or repeated final serving phase: ' + phase)
            return sha(controller)
        if state != previous:
            notify(state)
            previous = state
        if time.monotonic() >= deadline:
            raise TimeoutError('Study preparation wait expired; all model workers are preserved')
        time.sleep(min(poll, max(0., deadline - time.monotonic())))


def commits_in(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('commit', 'code_sha', 'source_commit') and isinstance(item, str) and re.fullmatch('[0-9a-f]{40}', item):
                found.add(item)
            found.update(commits_in(item))
    elif isinstance(value, list):
        for item in value:
            found.update(commits_in(item))
    return found


def source_commits(root, sources):
    """Inventory actual source versions, including prior exact/hash controls."""
    files = {root / 'quality-ready.json'}
    files.update(root.glob('*.json'))
    files.update(root.glob('cache-preflight-*/report.json'))
    files.update(root / family / 'report.json' for family in ('long-row-audit', 'llm-row-audit'))
    for family in ('long-frontier', 'llm-frontier'):
        prepared = load(root / family / 'prepare.json')
        for model in prepared['models']:
            for path in model['jobs']:
                job = load(Path(path)); files.add(Path(job['report']))
                parent = Path(job['report']).parent
                files.update(parent.glob(job['name'] + '-cache*-audit.json'))
            if family == 'llm-frontier':
                folder = root / family / model['name']
                files.update(folder.glob('*-ledger.json'))
                base, seed = model['name'].split('-exact-s')
                files.update(sources / 'runs' / f'{base}-20877-{kind}-s{seed}' / 'report.json'
                             for kind in ('exact', 'hash'))
    files.update((root / 'independent-controls').glob('*/report.json'))
    for phase in ('retime-long', 'retime-llm-cpu', 'retime-llm-gpu'):
        files.add(root / phase / 'report.json')
        files.update((root / phase).glob('[0-9][0-9][0-9][0-9][0-9].json'))
    commits = set()
    for path in sorted(files):
        commits.update(commits_in(load(path)))
    commits.add(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    for commit in sorted(commits):
        subprocess.run(['git', 'cat-file', '-e', commit + '^{commit}'], check=True)
    return sorted(commits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'sources', 'output', 'path-map'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-url', required=True)
    parser.add_argument('--wait-seconds', type=int, default=72 * 3600)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.wait_seconds < 1 or not 1 <= args.poll_seconds <= 60:
        parser.error('Positive wait and polling interval within 1–60 seconds required')
    if args.output.exists():
        parser.error('Fresh preparation output required; preserve prior output for diagnosis')
    args.output.mkdir(parents=True)
    report = {'status': 'waiting-for-final-serving', 'started': time.time(), 'stages': [],
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    def save():
        path = args.output / 'report.json'; pending = path.with_suffix('.pending')
        pending.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n'); pending.replace(path)
    def notify(state):
        report['serving_state'] = state; save(); print(state, flush=True)
    def run(name, module, arguments):
        command = [sys.executable, '-m', module, *map(str, arguments)]
        stage = {'name': name, 'command': command, 'started': time.time()}
        report['status'] = 'running-' + name; report['stages'].append(stage); save()
        with (args.output / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        stage.update(returncode=result.returncode, finished=time.time()); save()
        if result.returncode:
            raise RuntimeError('Evidence preparation failed; retained stage log: ' + name)
    save()
    try:
        report['final_serving_controller_sha256'] = wait_for_serving(
            args.root, args.wait_seconds, args.poll_seconds, notify)
        commits = source_commits(args.root, args.sources); report['runtime_commits'] = commits; save()
        evidence = args.output / 'evidence'
        run('publish', 'tools.publish_memory_application', [
            '--root', args.root, '--sources', args.sources, '--output', evidence,
            '--path-map', args.path_map, '--commits', *commits])
        run('verify-before-analysis', 'tools.verify_memory_application', ['--root', evidence])
        run('analyze', 'tools.analyze_memory_application', ['--evidence', evidence, '--output', evidence / 'analysis'])
        run('plot', 'tools.plot_memory_application', ['--evidence', evidence, '--output', evidence / 'figures'])
        run('package', 'tools.package_memory_application', [
            '--evidence', evidence, '--output', args.output / 'capsules', '--release-url', args.release_url])
        run('verify-final', 'tools.verify_memory_application', ['--root', evidence])
        if sha(args.root / 'final-serving-dispatch.json') != report['final_serving_controller_sha256']:
            raise ValueError('Completed serving controller changed during preparation')
        report.update(status='complete', finished=time.time(),
                      manifest_sha256=sha(args.output / 'capsules' / 'raw-manifest.json'))
    except BaseException as error:
        report.update(status='failed', error=repr(error), finished=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
