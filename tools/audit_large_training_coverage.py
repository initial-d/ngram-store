"""Separate actual full-corpus participation from changed-FP32-byte coverage.

Post-primary sensitivity: preserve every original qualification/failure and
freeze an alternative only on consistently rescored calibration documents.
Never edits training endpoints or selects using official test outcomes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np

from .large_application_frontier import publish_job
from .tree_node_quality import digest, load, write


def prepare(args):
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    data = args.root / 'data/dbpedia_14'; metadata = load(data / 'prepared.json')
    counts = np.load(data / 'counts.npy', mmap_mode='r')
    ids = np.load(data / 'train-ids.npy', mmap_mode='r')
    # Count the ACTUAL training event stream, not a recreated key fixture.
    if ids.min() < 1 or ids.max() > len(counts):
        raise ValueError('training ID out of registered domain')
    actual = np.bincount(np.asarray(ids), minlength=len(counts) + 1)
    if actual[0] or not np.array_equal(actual[1:], counts) or np.any(counts <= 0):
        raise ValueError('actual training participation differs from declared counts')
    if len(ids) != metadata['training_occurrences'] or len(counts) + 1 != metadata['rows']:
        raise ValueError('registered source size changed')
    write(output / 'participation.json', {'status': 'verified', 'rows': len(counts) + 1,
        'nonpadding_rows_observed': int(np.count_nonzero(actual[1:])), 'occurrences_per_epoch': len(ids),
        'minimum_occurrences_per_epoch': int(counts.min()),
        'counts_sha256': digest(data / 'counts.npy'), 'actual_train_ids_sha256': digest(data / 'train-ids.npy'),
        'prepared_sha256': digest(data / 'prepared.json'), 'source_sha256': digest(__file__),
        'boundary': 'All registered nonpadding IDs occur in the complete training stream. Participation does not establish a nonzero gradient, an FP32 change, or downstream necessity for every row.'})
    jobs = []
    paths = sorted((args.root / 'training-registered').glob('dbpedia_14-exact-uniform-*/training.json'))
    if len(paths) != 9: raise ValueError('complete three-seed/three-rate family required')
    for path in paths:
        report = load(path); source = path.parent
        if (report['status'] not in ('complete', 'incomplete-row-coverage') or len(report['epochs']) != 10
                or any(e['documents'] != 504000 for e in report['epochs'])
                or report['rows'] != len(counts) + 1 or report['settings']['epochs'] != 10):
            raise ValueError('source did not complete registered full-corpus schedule')
        if report['source_sha256'] != digest(Path(__file__).with_name('large_ngram_application.py')):
            raise ValueError('training loop implementation changed; manual audit required')
        files = {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in
            (source / 'rows.npy', source / 'classifier.npz', path,
             data / 'registry.npy', data / 'word-vocabulary.json', data / 'prepared.json')}
        job = {'name': source.name, 'source': str(source), 'data': str(data), 'kind': 'source',
            'dim': report['dim'], 'rows': report['rows'], 'files': files,
            'complete_model_bytes': sum(v['bytes'] for v in files.values()),
            'source_report_sha256': digest(path), 'quality_reference': 'source',
            'decoded_sha256': None, 'source_sha256': digest(__file__),
            'sensitivity': {'original_qualification': report['status'], 'changed_rows': report['changed_rows'],
                'participation_sha256': digest(output / 'participation.json'), 'learning_rate': report['settings']['lr'],
                'seed': report['seed'], 'test_used_for_selection': False}}
        job_path = output / 'jobs' / (source.name + '.json'); publish_job(job_path, job); jobs.append(str(job_path))
    write(output / 'prepared.json', {'status': 'complete', 'jobs': jobs,
        'protocol_sha256': digest(args.protocol), 'source_sha256': digest(__file__)})


def run(args):
    output = args.output; cfg = load(output / 'prepared.json')
    def evaluate(path, split, core):
        path = Path(path); result = output / 'quality' / (path.stem + '-' + split + '.json')
        if result.exists() and load(result).get('status') == 'complete': return str(result)
        result.parent.mkdir(parents=True, exist_ok=True)
        argv = ['taskset', '-c', str(core), sys.executable, '-m', 'tools.large_application_frontier',
                'evaluate', '--job', str(path), '--split', split, '--output', str(result)]
        with result.with_suffix('.log').open('w') as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
        return str(result)
    # All nine use identical FP64 reductions, including the six earlier passes.
    with ThreadPoolExecutor(max_workers=3) as pool:
        calibration = list(pool.map(lambda pair: evaluate(pair[1], 'calibration', 6 + pair[0] % 3), enumerate(cfg['jobs'])))
    by_rate = {}
    for path in cfg['jobs']:
        job = load(path); sensitivity = job['sensitivity']; rate = sensitivity['learning_rate']
        score = load(output / 'quality' / (Path(path).stem + '-calibration.json'))
        by_rate.setdefault(rate, []).append((path, job, score))
    if sorted(by_rate) != [1., 3., 10.] or any(sorted(v[1]['sensitivity']['seed'] for v in group) != [3, 7, 11] for group in by_rate.values()):
        raise ValueError('incomplete rescored calibration family')
    rate = min(by_rate, key=lambda lr: (np.mean([v[2]['mean_nll'] for v in by_rate[lr]]), lr))
    selected = [v[0] for v in by_rate[rate]]
    frozen = {'status': 'frozen', 'learning_rate': rate, 'selected': selected,
        'calibration_sha256': {p: digest(p) for p in calibration}, 'test_used': False,
        'mean_calibration_nll': {str(lr): float(np.mean([v[2]['mean_nll'] for v in group])) for lr, group in by_rate.items()},
        'source_sha256': digest(__file__), 'protocol_sha256': cfg['protocol_sha256'],
        'boundary': 'Post-primary participation sensitivity, not a replacement of registered byte-change-qualified results. All row-byte failures remain failures under the original rule.'}
    write(output / 'selection.json', frozen)
    with ThreadPoolExecutor(max_workers=3) as pool:
        test = list(pool.map(lambda pair: evaluate(pair[1], 'test', 6 + pair[0]), enumerate(selected)))
    write(output / 'complete.json', {'status': 'complete', 'selection_sha256': digest(output / 'selection.json'),
        'test_sha256': {p: digest(p) for p in test}, 'source_sha256': digest(__file__),
        'test_accuracy_mean': float(np.mean([load(p)['accuracy'] for p in test])),
        'test_nll_mean': float(np.mean([load(p)['mean_nll'] for p in test]))})


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='phase', required=True)
    prepare_parser = sub.add_parser('prepare')
    for name in ('root', 'output', 'protocol'): prepare_parser.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('run').add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.phase == 'prepare': prepare(args)
    else: run(args)


if __name__ == '__main__':
    main()
