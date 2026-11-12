"""Resume frozen Qwen controls as complete, independently scheduled paired blocks.

Each source/codec/cohort contains four rotated methods and both registered batch
sizes. Incomplete or mixed-affinity history is retained, never mixed into a new
pair. No calibration selection, weights or benchmark parameters are changed.
"""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .tree_node_quality import digest, load, write


def paired_cells(prepared, selection, output):
    entries = prepared['entries']; blocks = []
    for chosen in selection['selected']:
        codec = chosen['codec']
        reference = next(e for e in entries if e['codec'] == codec and e['variant'] == 'direct')
        matching = [e for e in entries if e['codec'] == codec
            and e.get('block_rows') == chosen['block_rows'] and e.get('predictor') == chosen['predictor']]
        variants = [reference, *matching]
        if len(variants) != 4 or {e['variant'] for e in variants} != {'direct', 'prior', 'zlib', 'libdeflate'}:
            raise ValueError('registered four-method paired control block required')
        for cohort in range(3):
            cells = []
            for entry in variants[cohort:] + variants[:cohort]:
                for batch in (1, 4):
                    name = f"{codec}-cohort{cohort}-batch{batch}-{Path(entry['job']).stem}"
                    cells.append({'job': entry['job'], 'batch_size': batch,
                                  'output': str(output / 'gpu' / (name + '.json'))})
            blocks.append({'name': f'{output.name}-{codec}-cohort{cohort}', 'source': output.name,
                'codec': codec, 'cohort': cohort, 'selection': str(output / 'selection.json'),
                'selection_sha256': digest(output / 'selection.json'), 'cells': cells})
    return blocks


def reusable_block(block):
    """A whole clean block is reusable; partial pairs never cross devices/cores."""
    reports = [Path(c['output']) for c in block['cells']]
    if not all(p.exists() for p in reports): return False
    values = [load(p) for p in reports]
    if any(v.get('status') != 'passed' for v in values): return False
    identities = {(v.get('measurement_host'), v.get('gpu_before', {}).get('gpu_uuid'),
        tuple(v.get('cpu_affinity', [])), v.get('phase_source_sha256')) for v in values}
    if len(identities) != 1: return False
    host, uuid, cores, source = next(iter(identities))
    if not host or not uuid or not source or len(cores) != 1: return False
    # Prior multi-core/SMT-overlap timing stays in the historical archive.
    if cores[0] not in range(32): return False
    for cell, value in zip(block['cells'], values):
        if value.get('job_sha256') != digest(cell['job']): return False
        if value.get('batch_size') != cell['batch_size']: return False
        if value.get('gpu_before', {}).get('external_compute_pids'): return False
        if any(s.get('external_compute_pids') for s in value.get('gpu_samples', [])): return False
    return True


def run_block(args):
    block = load(args.block)
    if digest(block['selection']) != block['selection_sha256']:
        raise ValueError('frozen selection changed')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    progress = args.block.with_suffix('.progress.json'); done = []
    for cell in block['cells']:
        out = Path(cell['output']); out.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'tools.benchmark_llm_codec_serving', 'gpu',
            '--job', cell['job'], '--output', str(out), '--upstream', str(args.upstream),
            '--requests', '8', '--repeat', '3', '--warmup', '2', '--decode-steps', '16',
            '--batch-size', str(cell['batch_size']), '--skip-spare-routing']
        with out.with_suffix('.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if load(out)['status'] != 'passed': raise RuntimeError('paired cell did not pass')
        done.append(str(out))
        write(progress, {'status': 'running', 'completed': done, 'expected': len(block['cells'])})
    if not reusable_block(block): raise RuntimeError('whole-block device/core/byte audit failed')
    write(progress, {'status': 'complete', 'completed': done, 'expected': len(block['cells']),
        'report_sha256': {p: digest(p) for p in done}})


def run(args):
    root = args.root; output = root / 'native-parallel'; output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'dispatch.json'
    if state_path.exists() and load(state_path).get('status') == 'running':
        pid = load(state_path).get('controller_pid')
        if pid and Path(f'/proc/{pid}/cmdline').exists(): raise RuntimeError('parallel controller already active')
    blocks = []
    for selection in sorted((root / 'native-serving').glob('*/selection.json')):
        prepared = root / 'native-layouts' / selection.parent.name / 'prepared.json'
        blocks.extend(paired_cells(load(prepared), load(selection), selection.parent))
    if len(blocks) != 108: raise ValueError('six sources, six codecs, three cohorts required')
    pending = []; completed = {}; failed = {}; running = {}
    archive = output / 'history' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for block in blocks:
        path = output / 'blocks' / (block['name'] + '.json'); write(path, block)
        if reusable_block(block):
            completed[block['name']] = {'status': 'reused-complete-paired-block',
                'reports': {c['output']: digest(c['output']) for c in block['cells']}}
            continue
        for cell in block['cells']:
            out = Path(cell['output'])
            for old in (out, out.with_suffix('.npz'), out.with_suffix('.log')):
                if old.exists():
                    dest = archive / block['source'] / old.name; dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old), dest)
        pending.append((block, path))
    # Start long-model blocks first; all choices are already frozen.
    pending.sort(key=lambda p: (not p[0]['source'].startswith('qwen35'), p[0]['name']))
    def save(status='running'):
        write(state_path, {'status': status, 'controller_pid': os.getpid(), 'reserved_gpus': list(range(8)) if status == 'running' else [],
            'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': digest(__file__),
            'running': {n: {k: v for k, v in x.items() if k not in ('child', 'log')} for n, x in running.items()},
            'completed': completed, 'failed': failed, 'pending_blocks': len(pending), 'expected_blocks': len(blocks),
            'pairing': 'Whole source/codec/cohort block, one physical GPU and one physical CPU core; no cross-device latency comparison.',
            'history': str(archive)})
    save()
    while pending or running:
        for name, item in list(running.items()):
            code = item['child'].poll()
            if code is None: continue
            item['log'].close(); final = {k: v for k, v in item.items() if k not in ('child', 'log')}
            final['exit_code'] = code
            (completed if code == 0 else failed)[name] = final
            del running[name]
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        occupied = {line.split(',')[0].strip() for line in raw.splitlines() if ',' in line}
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        busy = {x['gpu'] for x in running.values()}
        for line in devices.splitlines():
            index, uuid, util = [x.strip() for x in line.split(',')]; gpu = int(index)
            if not pending or gpu in busy or uuid in occupied or int(util) > 5: continue
            block, path = pending.pop(0); logpath = output / 'logs' / (block['name'] + '.log')
            logpath.parent.mkdir(parents=True, exist_ok=True); log = logpath.open('a')
            command = ['taskset', '-c', str(gpu if gpu < 6 else gpu + 20), sys.executable,
                '-m', 'tools.run_native_gpu_blocks', 'block', '--block', str(path), '--upstream', str(args.upstream)]
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            running[block['name']] = {'child': child, 'log': log, 'pid': child.pid, 'gpu': gpu,
                'command': command, 'block': str(path), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            save()
        save(); time.sleep(5)
    save('failed' if failed else 'complete')
    for selection in sorted((root / 'native-serving').glob('*/selection.json')):
        model_blocks = [b for b in blocks if b['source'] == selection.parent.name]
        if all(reusable_block(b) for b in model_blocks):
            write(selection.parent / 'complete.json', {'status': 'complete', 'selection_sha256': digest(selection),
                'source_sha256': digest(__file__), 'pairing_unit': 'source/codec/cohort',
                'gpu_reports': {c['output']: digest(c['output']) for b in model_blocks for c in b['cells']}})
    if failed: raise RuntimeError('paired blocks failed; history retained')


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='phase', required=True)
    c = sub.add_parser('run'); c.add_argument('--root', type=Path, required=True)
    c.add_argument('--upstream', type=Path, required=True)
    c = sub.add_parser('block'); c.add_argument('--block', type=Path, required=True)
    c.add_argument('--upstream', type=Path, required=True)
    args = p.parse_args(); (run if args.phase == 'run' else run_block)(args)


if __name__ == '__main__': main()
