"""Freeze-check, independently retime and publish the complete CPU frontier.

Publication contains numeric evidence only. Source rows, lexical keys and corpus
files are deliberately outside the allowlist; raw private hashes are retained.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from .tree_node_quality import (BUDGETS, SPLITS, bootstrap, digest, environment,
    load, quality_gate, query_job, select, write)

FAMILIES = ('raw', 'dense', 'constant', 'native-identity', 'tree-datrie',
            'tree-reference', 'faiss', 'faiss-dedup')


def check_selection(root):
    prepare, frozen, complete = [load(root / name) for name in
        ('prepare.json', 'selection.json', 'complete.json')]
    assert prepare['status'] == complete['status'] == 'complete'
    assert complete['frozen_selection_sha256'] == digest(root / 'selection.json')
    assert complete['models'] == len(prepare['models']) == 18
    assert complete['requested_candidates'] == 3672 and complete['fresh_audit_workers'] == 7344
    models = []
    for model, selected in zip(prepare['models'], frozen['models']):
        assert model['name'] == selected['name'] and len(model['jobs']) == 204
        jobs = {load(path)['name']: (path, load(path)) for path in model['jobs']}
        builds = {name: load(job['report']) for name, (_, job) in jobs.items()}
        assert len(builds) == len(jobs) == 204
        assert all(digest(jobs[name][1]['report']) == h
                   for name, h in selected['candidate_report_sha256'].items())
        assert set(selected['candidate_report_sha256']) == set(builds)
        for choice in selected['choices']:
            candidates = [b for b in builds.values() if b['family'] == choice['family']]
            best = select(candidates, choice['budget'], choice['cached'])
            assert choice['name'] == (best['name'] if best else None)
            assert choice['model_bytes'] == (best['model_bytes'] if best else None)
        assert len(selected['choices']) == 2 * len(BUDGETS) * len(FAMILIES)
        for name, (path, job) in jobs.items():
            b = builds[name]
            assert b['status'] == 'complete' and b['job_sha256'] == digest(path)
            for cached in (0, 1):
                audit = Path(job['report']).with_name(name + f'-cache{cached}-datrie-rowcache0-audit.json')
                a = load(audit)
                assert a['status'] == 'complete' and a['job_sha256'] == digest(path)
        models.append((model, selected, jobs, builds))
    return models


def retime(args):
    models = check_selection(args.frontier)
    args.output.mkdir(parents=True, exist_ok=False)
    requests = []
    for model, selected, jobs, _ in models:
        pairs = {(c['name'], c['cached']) for c in selected['choices'] if c['name']}
        # Retain identical matched controls even if a smaller family member won.
        for name in ('raw-fp32', 'raw-fp16', 'raw-int8', 'dense-fp32',
                     'native-none-fp32', 'native-none-fp16', 'native-none-int8',
                     'constant-zero', 'constant-mean', 'constant-weighted-mean'):
            pairs.update((name, cached) for cached in (0, 1))
        for name, cached in sorted(pairs):
            path, job = jobs[name]
            for index in ('datrie', 'hash'):
                for cache in ((0, 1048576) if job['backend'] == 'native' else (0,)):
                    requests.append({'model': model['name'], 'name': name, 'job': path,
                        'cached': cached, 'index': index, 'cache': cache})
    random.Random(42).shuffle(requests)
    report = {'status': 'running', 'environment': environment(),
        'frozen_selection_sha256': digest(args.frontier / 'selection.json'),
        'workers': 1, 'requests': requests, 'complete': [], 'failures': [],
        'execution_schedule_sha256': hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()}
    write(args.output / 'report.json', report)
    for i, request in enumerate(requests):
        log = args.output / f'{i:04d}.log'
        with log.open('w') as f:
            result = subprocess.run([sys.executable, '-m', 'tools.finalize_tree_node_quality',
                'query', '--job', request['job'], '--cached', str(request['cached']),
                '--index', request['index'], '--cache', str(request['cache'])], stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            report['failures'].append({'request': request, 'returncode': result.returncode})
        else:
            job = load(request['job'])
            path = Path(job['report']).with_name(request['name'] +
                f'-cache{request["cached"]}-{request["index"]}-rowcache{request["cache"]}-retiming.json')
            report['complete'].append({'request': request, 'report_sha256': digest(path)})
        write(args.output / 'report.json', report)
        if i % 20 == 0:
            print(json.dumps({'retimed': i + 1, 'expected': len(requests), 'failures': len(report['failures'])}), flush=True)
    report['status'] = 'complete' if not report['failures'] else 'failed'
    write(args.output / 'report.json', report)
    assert report['status'] == 'complete'


def publish(args):
    models = check_selection(args.frontier)
    final = load(args.retiming / 'report.json')
    assert final['status'] == 'complete' and not final['failures']
    assert final['frozen_selection_sha256'] == digest(args.frontier / 'selection.json')
    analysis = load(args.analysis / 'report.json')
    training = load(args.mechanism / 'report.json')
    assert analysis['status'] == training['status'] == 'complete'
    assert len(analysis['complete']) == 108 and len(training['trajectories']) == 36
    assert analysis['source_factorial_report_sha256'] == digest(args.mechanism / 'report.json')
    args.output.mkdir(parents=True, exist_ok=False)
    replacements = sorted(((str(args.frontier.parent), '${RUN_ROOT}'),
        ('/llm-align/duyimin/ngram-cross-corpus-20261102/training', '${TRAINING_ROOT}'),
        ('/llm-align/duyimin/ngram-counterfact-capacity-20261106/env', '${ENV_ROOT}'),
        ('/llm-align/duyimin/meft-cpu-capacity-20261030/enwik8.zip', '${CORPUS_ZIP}')),
        key=lambda v: -len(v[0]))
    def public(value):
        if isinstance(value, dict):
            return {k: public(v) for k, v in value.items()}
        if isinstance(value, list):
            return [public(v) for v in value]
        if isinstance(value, str):
            for a, b in replacements:
                value = value.replace(a, b)
            if '/llm-align/' in value:
                raise ValueError('unredacted remote path outside explicit replacements')
        return value
    private_hashes = {}
    def archive(source, target):
        private_hashes[target] = digest(source)
        write(args.output / target, public(load(source)))
    for file in ('prepare.json', 'selection.json', 'complete.json'):
        archive(args.frontier / file, file)
    archive(args.retiming / 'report.json', 'retiming-controller.json')
    archive(args.mechanism / 'report.json', 'mechanism-training.json')
    archive(args.analysis / 'report.json', 'mechanism-analysis.json')
    retimings = {}
    for item in final['complete']:
        r = item['request']; job = load(r['job'])
        path = Path(job['report']).with_name(r['name'] + f'-cache{r["cached"]}-{r["index"]}-rowcache{r["cache"]}-retiming.json')
        assert digest(path) == item['report_sha256']
        retimings.setdefault(r['model'], []).append({'name': r['name'], **public(load(path))})
    intervals = []
    for model, selected, jobs, builds in models:
        folder = 'models/' + model['name']
        common = Path(next(iter(jobs.values()))[1]['common'])
        archive(common / 'metadata.json', folder + '/metadata.json')
        # The original frozen ledger omitted the common reader metadata file.
        # Add its actual private bytes to every candidate after selection: this
        # model-wide constant cannot change any minimum-byte chosen name. Keep
        # the original reports/selection immutable and disclose both ledgers.
        extra_metadata = {'bytes': (common / 'metadata.json').stat().st_size,
                          'sha256': digest(common / 'metadata.json')}
        write(args.output / folder / 'complete-cost.json', {
            'reader_metadata_file': extra_metadata,
            'adjusted_common_model_bytes': load(common / 'metadata.json')['common_model_bytes'] + extra_metadata['bytes'],
            'selection_name_invariant': 'Same positive metadata-byte constant added to every candidate in this model; frozen names and quality unchanged.'})
        grid = []
        blocks_path = args.output / folder / 'paired-blocks.jsonl.gz'
        with gzip.GzipFile(filename=str(blocks_path), mode='wb', mtime=0) as f:
            for name, (_, job) in jobs.items():
                audits, blocks, audit_hashes, block_hashes = {}, {}, {}, {}
                for cached in (0, 1):
                    a = Path(job['report']).with_name(name + f'-cache{cached}-datrie-rowcache0-audit.json')
                    audits[str(cached)] = public(load(a))
                    audit_hashes[str(cached)] = digest(a)
                    blocks[str(cached)] = {split: load(Path(job['report']).with_name(
                        name + f'-{split}-cache{cached}-blocks.json.gz')) for split in SPLITS}
                    block_hashes[str(cached)] = {split: digest(Path(job['report']).with_name(
                        name + f'-{split}-cache{cached}-blocks.json.gz')) for split in SPLITS}
                f.write((json.dumps({'name': name, 'blocks': blocks, 'private_block_sha256': block_hashes},
                    separators=(',', ':'), allow_nan=False) + '\n').encode())
                grid.append({'build': public(builds[name]), 'audits': audits,
                    'private_build_sha256': digest(job['report']), 'private_audit_sha256': audit_hashes,
                    'complete_serving_bytes': builds[name]['model_bytes'] + extra_metadata['bytes']})
        write(args.output / folder / 'grid.json.gz', grid)
        write(args.output / folder / 'retimings.json.gz', retimings[model['name']])
        pairs = sorted({(c['name'], c['cached']) for c in selected['choices'] if c['name']})
        # Bootstrap every unique frozen choice, without reselecting on either audit.
        selected_intervals = []
        for name, cached in pairs:
            job = jobs[name][1]
            for split in ('test', 'confirmation'):
                path = Path(job['report']).with_name(name + f'-{split}-cache{cached}-blocks.json.gz')
                blocks = load(path)
                selected_intervals.append({'name': name, 'cached': cached, 'split': split,
                    'private_paired_blocks_sha256': digest(path),
                    'strata': {s: bootstrap(blocks[s]) for s in ('all', 'oov_nonempty')}})
        write(args.output / folder / 'selected-intervals.json.gz', selected_intervals)
        intervals.append({'model': model['name'], 'frozen_unique_choices': len(pairs)})
    for t in training['trajectories']:
        trajectory = args.mechanism / t['name']
        assert digest(trajectory / 'report.json') == t['report_sha256']
        archive(trajectory / 'report.json', 'trajectories/' + t['name'] + '/report.json')
        schedule = trajectory.parent / 'schedule' / 'schedule.json'
        target = 'schedules/' + trajectory.parent.name + '.json'
        if not (args.output / target).exists():
            archive(schedule, target)
    for point in analysis['complete']:
        source = args.analysis / 'points' / point['name']
        assert digest(source / 'report.json') == point['report_sha256']
        for path in sorted(source.iterdir()):
            if path.is_file() and (path.name == 'report.json' or path.suffix == '.gz'):
                archive(path, 'points/' + point['name'] + '/' + path.name)
    write(args.output / 'publication.json', {'environment': public(environment()),
        'status': 'complete', 'models': 18, 'candidates': 3672, 'audit_workers': 7344,
        'mechanism_trajectories': 36, 'mechanism_points': 108, 'selected_intervals': intervals,
        'retiming_workers': 1, 'retiming_requests': len(final['requests']),
        'private_input_sha256': private_hashes,
        'redaction': 'Explicit environment path placeholders; numeric evidence and hashes unchanged. No keys, corpus or weights published.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('retime', 'publish', 'query'))
    for name in ('frontier', 'mechanism', 'analysis', 'retiming', 'output', 'job'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--cached', type=int, choices=(0, 1), default=0)
    parser.add_argument('--index', choices=('datrie', 'hash'), default='datrie')
    parser.add_argument('--cache', type=int, choices=(0, 1048576), default=0)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU only; empty CUDA_VISIBLE_DEVICES required')
    if args.phase == 'query':
        query_job(args.job, args.cached, index_kind=args.index, cache_bytes=args.cache, selected=True)
    elif args.phase == 'retime':
        retime(args)
    else:
        publish(args)


if __name__ == '__main__':
    main()
