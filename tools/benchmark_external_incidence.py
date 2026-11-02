"""Fresh CPU workers: exact external construction versus current fit/means."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import sqlite3
import subprocess
import sys
import time

import numpy as np
from .cotraining_groups import BagPartition
from .external_cotraining import file_hash, iter_bag_files


def disk_bytes(path):
    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            total += disk_bytes(entry.path) if entry.is_dir(follow_symlinks=False) else entry.stat().st_size
    return total


def compare_files(left, right):
    compared = 0
    with Path(left).open('rb') as a, Path(right).open('rb') as b:
        while True:
            x, y = a.read(65536), b.read(65536)
            if x != y:
                raise ValueError('output bytes differ: ' + str(left))
            if not x:
                return compared
            compared += len(x)


def worker(path):
    job = json.loads(Path(path).read_text())
    inputs = json.loads(Path(job['input']).read_text())
    out = Path(job['output'])
    out.mkdir(parents=True, exist_ok=False)
    for key in ('source', 'ids', 'offsets'):
        if file_hash(inputs[key]) != inputs[key + '_sha256']:
            raise ValueError('input hash mismatch: ' + key)
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'job': job,
              'input': inputs, 'python': sys.version, 'numpy': np.__version__,
              'sqlite': sqlite3.sqlite_version, 'os': platform.platform(), 'device': 'cpu',
              'source_files_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
                  ('benchmark_external_incidence.py', 'external_cotraining.py', 'cotraining_groups.py')},
              'initial_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
              (1 if sys.platform == 'darwin' else 1024)}
    started, peak_disk = time.perf_counter(), 0

    def progress(stage, directory):
        nonlocal peak_disk
        peak_disk = max(peak_disk, disk_bytes(out))
        if stage != 'sorted-run' and stage != 'merged-run':
            print(json.dumps({'stage': stage, 'elapsed_seconds': time.perf_counter() - started,
                              'workspace_logical_bytes': disk_bytes(out)}), flush=True)

    bags = iter_bag_files(inputs['ids'], inputs['offsets'])
    if job['method'] == 'legacy':
        partition = BagPartition.fit(bags, inputs['rows'])
        report['partition_seconds'] = time.perf_counter() - started
        before = time.perf_counter()
        source = np.memmap(inputs['source'], mode='r', dtype='<f4', shape=(inputs['rows'], inputs['dim']))
        means = partition.means(source)
        means.tofile(out / 'means.f32')
        report['projection_seconds'] = time.perf_counter() - before
        partition.mapping.astype('<u8').tofile(out / 'mapping.u64')
        partition.sizes.astype('<u8').tofile(out / 'sizes.u64')
        partition.save(out / 'domain.npz')
        report['projection'] = {'means_sha256': file_hash(out / 'means.f32'),
                                'source_sha256': inputs['source_sha256']}
        report['map_directory'] = str(out)
    else:
        budget = job['memory_mib'] * 1024**2
        partition = BagPartition.fit_external(bags, inputs['rows'], out / 'partition',
                                             memory_bytes=budget, progress=progress)
        report['partition_seconds'] = time.perf_counter() - started
        before = time.perf_counter()
        report['projection'] = partition.project_file(inputs['source'], inputs['dim'], out / 'means.f32',
                                                      memory_bytes=budget, progress=progress)
        report['projection_seconds'] = time.perf_counter() - before
        partition.save_npz(out / 'domain.npz')
        report['map_directory'] = str(out / 'partition')
    report.update(partition=partition.summary(), total_seconds=time.perf_counter() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                  (1 if sys.platform == 'darwin' else 1024),
                  sampled_peak_workspace_logical_bytes=max(peak_disk, disk_bytes(out)),
                  final_artifact_bytes=disk_bytes(out), status='completed')
    if file_hash(inputs['source']) != inputs['source_sha256'] or \
            report['projection']['source_sha256'] != inputs['source_sha256']:
        raise ValueError('source table changed during construction')
    if report['partition']['domain_sha256'] != inputs['domain_sha256']:
        raise ValueError('partition changed input domain')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'finished': job['method'], 'rows': inputs['rows'],
                      'peak_rss_bytes': report['peak_rss_bytes'], 'seconds': report['total_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--worker', type=Path)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    if cfg.worker:
        worker(cfg.worker)
        return
    if not cfg.inputs or not cfg.output_dir:
        p.error('inputs and output-dir required')
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((cfg.inputs / 'report.json').read_text())
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'command': shlex.join([sys.executable, *sys.argv]), 'preparation_report_sha256': file_hash(cfg.inputs / 'report.json'),
              'cpu': subprocess.check_output(['lscpu', '-J'], text=True), 'runs': [], 'failures': [],
              'limitations': ['Shared warm CPU/file cache; one fresh worker per build.',
                  'Managed memory is not a hard process RSS or OS page-cache cap.',
                  'Corpus/JSON preparation, runtime materialization and native packing are separate.',
                  'Workspace disk is sampled at runs/merges/stage boundaries, not a continuous physical-disk peak.',
                  'Only actual finite trained row prefixes; no new training or semantic/LLM quality claim.']}
    for dataset in inputs['sets']:
        full = dataset['rows'] == max(s['rows'] for s in inputs['sets'])
        jobs = [('legacy', None)] + [('external', budget) for budget in ((64, 128, 256) if full else (64,))]
        baseline = None
        for method, budget in jobs:
            name = f"{dataset['rows']}-{method}" + (f"-{budget}" if budget else '')
            out = cfg.output_dir / name
            job = {'method': method, 'memory_mib': budget,
                   'input': str(cfg.inputs / str(dataset['rows']) / 'input.json'), 'output': str(out)}
            job_file = cfg.output_dir / (name + '-job.json')
            job_file.write_text(json.dumps(job, indent=2) + '\n')
            with (cfg.output_dir / (name + '.log')).open('w') as log:
                result = subprocess.run([sys.executable, '-m', 'tools.benchmark_external_incidence',
                                         '--worker', str(job_file)], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                report['failures'].append({'name': name, 'exit_code': result.returncode})
            else:
                run = json.loads((out / 'report.json').read_text())
                if method == 'legacy':
                    baseline = run
                else:
                    if baseline is None or run['partition'] != baseline['partition']:
                        raise ValueError('missing baseline or changed group semantics')
                    run['all_output_bytes_identical'] = {
                        'mapping': compare_files(Path(run['map_directory']) / 'mapping.u64',
                                                 Path(baseline['map_directory']) / 'mapping.u64'),
                        'sizes': compare_files(Path(run['map_directory']) / 'sizes.u64',
                                               Path(baseline['map_directory']) / 'sizes.u64'),
                        'means': compare_files(out / 'means.f32', Path(baseline['job']['output']) / 'means.f32')}
                report['runs'].append(run)
                print(json.dumps({'completed': name, 'seconds': run['total_seconds'],
                                  'peak_rss_bytes': run['peak_rss_bytes']}), flush=True)
            (cfg.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'completed' if not report['failures'] else 'completed-with-failures'
    (cfg.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
