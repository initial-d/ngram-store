"""Retry only externally interrupted GPU measurements; retain every failed run."""
import json
import hashlib
import subprocess
import time
from pathlib import Path


def competition(error):
    return ('GPU competition before model loading' in error
            or 'GPU competition appeared during serving' in error)


def wait_idle(selector, deadline):
    uuid = subprocess.check_output(['nvidia-smi', '--id=' + selector, '--query-gpu=uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    while time.monotonic() < deadline:
        clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        occupied = any(line.split(',')[0].strip() == uuid for line in clients.splitlines() if ',' in line)
        utilization = int(subprocess.check_output(['nvidia-smi', '--id=' + uuid,
            '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
        if not occupied and utilization <= 5: return
        time.sleep(5)
    raise TimeoutError('GPU idle wait expired; competing processes preserved')


def retain_interruption(output, log, error):
    event = {'time': time.time(), 'error': error, 'log': str(log),
             'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None}
    if output.exists():
        target = output.with_name(output.stem + '.interrupted-' + str(time.time_ns()) + '.json')
        event.update(output=str(target), sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        output.rename(target)
    with output.with_name(output.stem + '.interruptions.jsonl').open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True) + '\n')


def run_gpu_job(command, *, env, output, log, timeout=24 * 3600):
    output, log = Path(output), Path(log)
    deadline = time.monotonic() + timeout
    if output.exists():
        value = json.loads(output.read_text())
        if value.get('status') == 'passed': return subprocess.CompletedProcess(command, 0)
        if not competition(value.get('error', '')):
            raise ValueError('Existing failed GPU result needs diagnosis; only competition is retried')
        retain_interruption(output, log, value['error'])
    while True:
        with log.open('a') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode == 0: return result
        value = json.loads(output.read_text()) if output.exists() else {}
        error = value.get('error', '')
        if not competition(error):
            error = log.read_text()[-16000:]
            if not competition(error): return result
        retain_interruption(output, log, error)
        wait_idle(env['CUDA_VISIBLE_DEVICES'], deadline)
        log = log.with_name(log.stem + '.retry-' + str(time.time_ns()) + '.log')
