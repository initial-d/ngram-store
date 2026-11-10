"""Complete finite trained-table frontier; model quality is a separate phase."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .tree_node_quality import (BUDGETS, GATES, Reader, array_digest, digest,
    environment, execute_jobs, load, prepare_model, select, write)


def execute_resumable(jobs, phase, workers, root):
    """Keep completed hash-matched workers and preserve every failed attempt."""
    root.mkdir(parents=True, exist_ok=True)
    progress = root / (phase + '-progress.json')
    previous = load(progress) if progress.exists() else {}
    failures = list(previous.get('failures', []))
    requested = [(str(p), cached) for p in jobs for cached in ((None,) if phase == 'build' else (0, 1))]
    np.random.default_rng(42).shuffle(requested)
    report = {'phase': phase, 'jobs': len(jobs), 'complete': [], 'failures': failures,
        'execution_schedule_sha256': hashlib.sha256(json.dumps(requested).encode()).hexdigest(),
        'attempt_started': time.time()}
    def run(pair):
        path, cached = pair; job = load(path)
        target = Path(job['report']) if phase == 'build' else Path(job['report']).with_name(
            job['name'] + f'-cache{cached}-datrie-rowcache0-audit.json')
        if target.exists():
            existing = load(target)
            if existing.get('status') == 'complete' and existing.get('job_sha256') == digest(path):
                files = existing.get('serving_files', {})
                if any(digest(Path(job['directory']) / name) != entry['sha256'] for name, entry in files.items()):
                    raise ValueError('completed serving bytes changed')
                return {'job': path, 'cached': cached, 'report_sha256': digest(target), 'reused_completed_worker': True}
            raise ValueError('existing worker is incomplete or has a different job hash')
        if phase == 'build' and Path(job['directory']).exists():
            # Own run-directory output only. Keep partial bytes as failed evidence.
            Path(job['directory']).rename(str(job['directory']) + '.failed-' + str(time.time_ns()))
        log = root / (hashlib.sha256(json.dumps(pair).encode()).hexdigest() + '-' + str(time.time_ns()) + '.log')
        command = [sys.executable, '-m', 'tools.tree_node_quality', phase, '--job', path]
        if phase == 'audit': command.append('--quality-only')
        if cached is not None: command += ['--cached', str(cached)]
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{path}: exit {result.returncode}; {log}')
        value = load(target)
        if value['status'] != 'complete' or value['job_sha256'] != digest(path):
            raise ValueError('worker did not write complete hash-matched evidence')
        return {'job': path, 'cached': cached, 'report_sha256': digest(target), 'log': str(log)}
    current_failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run, pair): pair for pair in requested}
        for future in as_completed(pending):
            try: report['complete'].append(future.result())
            except Exception as error:
                entry = {'job': pending[future], 'error': repr(error), 'time': time.time()}
                current_failures.append(entry); report['failures'].append(entry)
            if len(report['complete']) % 20 == 0 or current_failures: write(progress, report)
            if len(report['complete']) % 20 == 0 or current_failures:
                print(json.dumps({'phase': phase, 'complete': len(report['complete']),
                    'expected': len(requested), 'current_failures': len(current_failures)}), flush=True)
    report['status'] = 'complete' if not current_failures and len(report['complete']) == len(requested) else 'failed'
    write(progress, report)
    if report['status'] != 'complete': raise RuntimeError('incomplete finite worker grid')


def add_native_code_jobs(model):
    """Every declared codec gets all three exact-code storage compositions."""
    jobs = list(model['jobs'])
    direct = [p for p in jobs if load(p)['backend'] == 'faiss']
    for path in direct:
        original = load(path)
        dedup_path = Path(path).with_name('dedup-' + original['encoding'] + '.json')
        dedup = load(dedup_path)
        for family, prefix, source, planner in (
            ('faiss-native', 'native-code-', original, 'none'),
            ('faiss-native-layout', 'native-code-layout-', original, 'layout'),
            ('faiss-dedup-native', 'native-code-dedup-', dedup, 'none'),
        ):
            name = prefix + original['encoding']
            job = {**source, 'name': name, 'family': family, 'planner': planner,
                'native_codes_source': source['directory'],
                'native_codes_source_report': source['report'],
                'directory': str(Path(source['directory']).parent / name),
                'report': str(Path(source['report']).with_name(name + '.json'))}
            target = Path(path).with_name(name + '.json')
            write(target, job); jobs.append(str(target))
    if len(jobs) != 261:
        raise AssertionError('complete 261-candidate grid required')
    write(Path(jobs[0]).parent.parent / 'jobs.json', jobs)
    return {**model, 'jobs': jobs}


def prepare_point(path):
    import torch
    from .train_corpus_checkpoints import corpus_prefix
    from .benchmark_heldout_compression import split_tokens
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    job = load(path); checkpoint_path = Path(job['checkpoint'])
    if digest(checkpoint_path) != job['point']['checkpoint_sha256']:
        raise ValueError('trained checkpoint changed')
    trajectory_path = Path(job['trajectory_report'])
    if digest(trajectory_path) != job['trajectory_report_sha256']:
        raise ValueError('trajectory report changed')
    original = Path(job['training_root']) / 'corpus.txt'
    training = load(original.parent / 'report.json')
    if digest(original) != training['tokenized_corpus_sha256']:
        raise ValueError('original training corpus changed')
    protocol = load(job['protocol'])['long_training']
    if digest(job['corpus_zip']) != training['corpus_zip_sha256']:
        raise ValueError('corpus ZIP changed')
    final = protocol['new_confirmation_token_interval'][1]
    full, raw_hash = corpus_prefix(Path(job['corpus_zip']), 'enwik8', final, protocol['raw_prefix_bytes'])
    tokens = full.split()
    if len(tokens) != final or ' '.join(tokens[:200000]) != original.read_text():
        raise ValueError('new held-out tokenization is not the original prefix extension')
    split = {'train': split_tokens(original.read_text())['train']}
    for name in ('validation', 'test', 'confirmation'):
        a, b = protocol['new_' + name + '_token_interval']; split[name] = tokens[a:b]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if array_digest(checkpoint['input_rows'].numpy()) != job['point']['input_row_sha256']:
        raise ValueError('trained input bytes changed')
    if array_digest(checkpoint['output_rows'].numpy()) != job['point']['output_row_sha256']:
        raise ValueError('trained output bytes changed')
    del checkpoint
    expected = {'convention': job['convention'], 'checkpoint_sha256': job['point']['checkpoint_sha256']}
    model = prepare_model(checkpoint_path, training, split, Path(job['output']),
        Path(job['native_library']), Path(job['structural_library']), expected=expected)
    common = Path(job['output']) / 'common'
    meta = load(common / 'metadata.json')
    meta.update(source_trajectory_report_sha256=job['trajectory_report_sha256'],
        point=job['point'], initialization_scale=job['scale'], raw_prefix_sha256=raw_hash,
        study_protocol_sha256=digest(job['protocol']), category='trained SGNS duration/initialization frontier')
    write(common / 'metadata.json', meta)
    model = add_native_code_jobs(model)
    model['metadata_sha256'] = digest(common / 'metadata.json')
    write(Path(job['output']) / 'prepared.json', model)


def prepare_all(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    training = load(args.mechanism / 'report.json')
    if training['status'] != 'complete' or len(training['trajectories']) != 36:
        raise ValueError('complete original factorial required')
    requests = []
    for item in training['trajectories']:
        folder = args.mechanism / item['name']; trajectory = load(folder / 'report.json')
        if digest(folder / 'report.json') != item['report_sha256'] or trajectory['status'] != 'complete':
            raise ValueError('trajectory source changed')
        for point in trajectory['checkpoints']:
            name = item['name'].replace('/', '-') + '-step-' + str(point['steps'])
            job = {'checkpoint': str(folder / ('step-' + str(point['steps'])) / 'reference.pt'),
                'point': point, 'convention': trajectory['convention'], 'scale': trajectory['ngram_initialization_scale'],
                'trajectory_report': str(folder / 'report.json'), 'trajectory_report_sha256': item['report_sha256'],
                'training_root': str(args.training_root), 'protocol': str(args.protocol),
                'corpus_zip': str(args.corpus_zip), 'native_library': str(args.native_library),
                'structural_library': str(args.structural_library), 'output': str(root / name)}
            path = root / 'preparation-jobs' / (name + '.json')
            if path.exists() and load(path) != job: raise ValueError('preparation request changed')
            write(path, job); requests.append(path)
    if len(requests) != 108:
        raise AssertionError('108 trained points required')
    prior = load(root / 'prepare.json') if (root / 'prepare.json').exists() else {}
    report = {'status': 'running', 'environment': environment(), 'protocol_sha256': digest(args.protocol),
        'source_factorial_sha256': digest(args.mechanism / 'report.json'), 'requested_points': 108,
        'models': [], 'failures': [], 'previous_failures': prior.get('previous_failures', []) + prior.get('failures', [])}
    write(root / 'prepare.json', report)
    def run(path):
        output = Path(load(path)['output']); completed = output / 'prepared.json'
        if completed.exists():
            model = load(completed)
            if model['metadata_sha256'] != digest(output / 'common' / 'metadata.json') or len(model['jobs']) != 261:
                raise ValueError('completed preparation changed')
            return model
        if output.exists(): output.rename(str(output) + '.failed-' + str(time.time_ns()))
        with path.with_suffix('.' + str(time.time_ns()) + '.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'tools.complete_memory_frontier',
                'prepare-point', '--job', str(path)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(str(path) + ': exit ' + str(result.returncode))
        return load(Path(load(path)['output']) / 'prepared.json')
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run, path): path for path in requests}
        for future in as_completed(pending):
            try: report['models'].append(future.result())
            except Exception as error: report['failures'].append({'job': str(pending[future]), 'error': repr(error)})
            write(root / 'prepare.json', report)
            print(json.dumps({'prepared': len(report['models']), 'requested': 108, 'failures': len(report['failures'])}), flush=True)
    report['models'].sort(key=lambda x: x['name'])
    report['status'] = 'complete' if len(report['models']) == 108 and not report['failures'] else 'failed'
    write(root / 'prepare.json', report)
    if report['status'] != 'complete': raise RuntimeError('complete preparation required')


def build_all(args):
    root = args.output; prepared = load(root / 'prepare.json')
    if prepared['status'] != 'complete': raise ValueError('complete preparation required')
    jobs = [p for m in prepared['models'] for p in m['jobs']]
    stages = (
        ('direct', [p for p in jobs if not load(p).get('native_codes_source') and load(p)['backend'] != 'dedup']),
        ('dedup', [p for p in jobs if not load(p).get('native_codes_source') and load(p)['backend'] == 'dedup']),
        ('native-codes', [p for p in jobs if load(p).get('native_codes_source')]),
    )
    for name, paths in stages:
        stage = root / ('stage-' + name)
        execute_resumable(paths, 'build', args.jobs, stage)
    freeze_all(args)


def freeze_all(args):
    root = args.output; prepared = load(root / 'prepare.json')
    frozen = {'environment': environment(), 'protocol_sha256': prepared['protocol_sha256'],
        'selection_uses': 'new validation only', 'models': []}
    for model in prepared['models']:
        costs, candidates = {}, []
        meta_file = Path(load(model['jobs'][0])['common']) / 'metadata.json'
        for path in model['jobs']:
            job = load(path); original = load(job['report'])
            if original['status'] != 'complete' or original['job_sha256'] != digest(path):
                raise ValueError('incomplete or changed candidate')
            complete_bytes = original['model_bytes'] + meta_file.stat().st_size
            candidates.append({**original, 'model_bytes': complete_bytes})
            costs[job['name']] = {'complete_serving_bytes': complete_bytes, 'raw_build_report_sha256': digest(job['report'])}
        families = sorted({r['family'] for r in candidates}); choices = []
        for cached in (0, 1):
            for budget in BUDGETS:
                for family in families:
                    selected = select([r for r in candidates if r['family'] == family], budget, cached)
                    choices.append({'cached': cached, 'budget': budget, 'family': family,
                        'name': selected['name'] if selected else None,
                        'complete_serving_bytes': selected['model_bytes'] if selected else None})
        write(meta_file.parent.parent / 'complete-cost.json', {'reader_metadata': {'bytes': meta_file.stat().st_size,
            'sha256': digest(meta_file)}, 'candidates': costs})
        frozen['models'].append({'name': model['name'], 'choices': choices, 'costs': costs})
    selection = root / 'selection.json'
    if selection.exists():
        previous = load(selection)
        if any(previous[key] != frozen[key] for key in ('protocol_sha256', 'selection_uses', 'models')):
            raise ValueError('existing frozen selection differs; preserve it and investigate')
    else: write(selection, frozen)
    # An independently trained source is chosen by absolute validation loss,
    # before any test/confirmation score is read. Compression qualification
    # remains relative to that selected source, with every fixed point retained.
    groups = {}
    for model in prepared['models']:
        common = Path(load(model['jobs'][0])['common']); meta = load(common / 'metadata.json')
        scores = dict(np.load(common / 'validation-source-scores.npz'))
        key = (meta['convention'], meta['order'], meta['seed'])
        groups.setdefault(key, []).append({'name': model['name'], 'steps': meta['point']['steps'],
            'scale': meta['initialization_scale'], 'absolute_validation_loss': float(np.mean(scores['loss'])),
            'absolute_validation_discrimination': float(np.mean(scores['discrimination'])),
            'source_scores_sha256': digest(common / 'validation-source-scores.npz')})
    training_selection = {'protocol_sha256': prepared['protocol_sha256'],
        'rule': 'Minimum absolute all-pair validation source loss, then higher discrimination, then stable name; test never selects',
        'groups': [{'convention': key[0], 'order': key[1], 'seed': key[2],
            'chosen': min(points, key=lambda p: (p['absolute_validation_loss'], -p['absolute_validation_discrimination'], p['name']))['name'],
            'points': sorted(points, key=lambda p: p['name'])} for key, points in sorted(groups.items())]}
    target = root / 'training-selection.json'
    if target.exists() and load(target) != training_selection: raise ValueError('existing training choice changed')
    write(target, training_selection)


def audit_all(args):
    root = args.output; prepared = load(root / 'prepare.json')
    frozen_hash = digest(root / 'selection.json')
    jobs = [p for m in prepared['models'] for p in m['jobs']]
    stage = root / 'stage-audit'
    execute_resumable(jobs, 'audit', args.jobs, stage)
    if digest(root / 'selection.json') != frozen_hash: raise AssertionError('frozen selection changed')
    write(root / 'quality-complete.json', {'status': 'complete', 'models': len(prepared['models']),
        'requested_candidates': len(jobs), 'fresh_audit_workers': len(jobs) * 2,
        'frozen_selection_sha256': frozen_hash})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'prepare-point', 'build', 'freeze', 'audit'))
    for name in ('output', 'mechanism', 'training-root', 'corpus-zip', 'native-library', 'structural-library', 'protocol', 'job'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--jobs', type=int, default=24)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': parser.error('CPU phases require empty CUDA_VISIBLE_DEVICES')
    if args.phase == 'prepare-point': prepare_point(args.job)
    elif args.phase == 'prepare': prepare_all(args)
    elif args.phase == 'build': build_all(args)
    elif args.phase == 'freeze': freeze_all(args)
    else: audit_all(args)


if __name__ == '__main__':
    main()
