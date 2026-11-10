"""Full post-quantization row fidelity, independently reopening every bundle."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .tree_node_quality import Reader, array_digest, digest, environment, load, write


def audit(path, output):
    job = load(path); common = Path(job['common'])
    source = np.load(common / 'source-rows.npy', mmap_mode='r')
    occurrence = np.load(common / 'source-occurrence-counts.npy')
    decoded = []
    reader = Reader(job)
    for start in range(0, len(source), 4096):
        ids = np.arange(start, min(start + 4096, len(source)))
        decoded.append(reader.rows(reader.mapping[ids] if reader.mapping is not None else ids))
    decoded = np.concatenate(decoded)
    build = load(job['report'])
    matched_build = array_digest(decoded.astype('<f4')) == build['decoded_row_sha256']
    construction_decoded = decoded
    rounding = None
    validation_one_thread = None
    validation_one_thread_blocks = None
    if not matched_build:
        # OPQ reverse transforms have thread-dependent FP32 matrix rounding.
        # Match the original four-thread build byte for byte, while auditing
        # the actual one-thread serving values rather than changing its mode.
        assert job.get('encoding', '').startswith('opq-')
        import faiss
        from .tree_node_quality import evaluate_hidden, score_candidate, quality_gate, BUDGETS
        for name, recorded in build['serving_files'].items():
            assert digest(Path(job['directory']) / name) == recorded['sha256']
        reader.close(); faiss.omp_set_num_threads(4); reader = Reader(job)
        construction = []
        for start in range(0, len(source), 4096):
            ids = np.arange(start, min(start + 4096, len(source)))
            construction.append(reader.rows(reader.mapping[ids] if reader.mapping is not None else ids))
        construction_decoded = np.concatenate(construction)
        assert array_digest(construction_decoded.astype('<f4')) == build['decoded_row_sha256']
        difference = decoded.astype(np.float64) - construction_decoded.astype(np.float64)
        rounding = {'construction_omp_threads': 4, 'query_omp_threads': 1,
            'maximum_absolute_difference': float(np.abs(difference).max()),
            'relative_frobenius_difference': float(np.linalg.norm(difference) / np.linalg.norm(construction_decoded.astype(np.float64))),
            'construction_reconstruction_byte_exact_to_build': True,
            'serialized_serving_files_sha256_verified': True}
        reader.close(); faiss.omp_set_num_threads(1); reader = Reader(job)
        validation_one_thread = {}
        validation_one_thread_blocks = {}
        changes = []
        for cached in (0, 1):
            q, blocks = score_candidate(evaluate_hidden(reader, common, cached), common, 'validation')
            validation_one_thread[str(cached)] = q
            validation_one_thread_blocks[str(cached)] = blocks
            for budget in BUDGETS:
                before, after = quality_gate(build['validation'][str(cached)], budget), quality_gate(q, budget)
                if before != after:
                    changes.append({'cached': cached, 'budget': budget, 'construction_pass': before, 'query_pass': after})
        rounding['validation_gate_changes'] = changes
    a, b = source.astype(np.float64), decoded.astype(np.float64)
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (an > 0) & (bn > 0)
    cosine = (a[valid] * b[valid]).sum(1) / (an[valid] * bn[valid])
    ratio = np.minimum(an[valid], bn[valid]) / np.maximum(an[valid], bn[valid])
    error = np.linalg.norm(a - b, axis=1) / np.maximum(an, 1e-30)
    result = {'environment': environment(), 'name': job['name'], 'rows': len(a),
        'phase_source_sha256': digest(__file__),
        'private_build_report_sha256': digest(job['report']), 'job_sha256': digest(path),
        'decoded_row_sha256': array_digest(decoded.astype('<f4')),
        'construction_decoded_row_sha256': array_digest(construction_decoded.astype('<f4')),
        'construction_relative_frobenius': float(np.linalg.norm(a - construction_decoded.astype(np.float64)) / np.linalg.norm(a)),
        'thread_rounding': rounding, 'validation_at_query_threads': validation_one_thread,
        'validation_at_query_threads_paired_blocks': validation_one_thread_blocks,
        'undefined_zero_norm_rows': int((~valid).sum()),
        'relative_l2': {'p50': float(np.median(error)), 'p95': float(np.quantile(error, .95)),
            'maximum': float(error.max()),
            'global': float(np.linalg.norm(a - b) / np.linalg.norm(a)),
            'occurrence_weighted': float(np.sqrt(np.sum(occurrence * np.square(a - b).sum(1)) /
                                              np.sum(occurrence * np.square(a).sum(1))))},
        'cosine': {'minimum': float(cosine.min()) if len(cosine) else None,
            'p50': float(np.median(cosine)) if len(cosine) else None,
            'p05': float(np.quantile(cosine, .05)) if len(cosine) else None},
        'norm_ratio': {'minimum': float(ratio.min()) if len(ratio) else None,
            'p50': float(np.median(ratio)) if len(ratio) else None},
        'source_row_bytes_preserved': a.tobytes() == b.tobytes(),
        'gates_on_original_rows': {'cosine': job.get('cosine'), 'norm_ratio': job.get('norm_ratio')},
        'post_encoding_rows_outside_original_joint_gate': (int(np.sum((cosine < job['cosine'] - 1e-7) |
            (ratio < job['norm_ratio'] - 1e-7))) + int((~valid).sum())) if job.get('cosine') else None,
        'status': 'complete'}
    reader.close()
    write(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontier', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--jobs', type=int, default=24)
    parser.add_argument('--expected-candidates', type=int, default=3672)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU only; empty CUDA_VISIBLE_DEVICES required')
    if args.job:
        import faiss
        faiss.omp_set_num_threads(1)
        audit(args.job, args.output); return
    selection_hash = digest(args.frontier / 'selection.json')
    args.output.mkdir(parents=True, exist_ok=True)
    prior = load(args.output / 'report.json') if (args.output / 'report.json').exists() else {}
    report = {'environment': environment(), 'frozen_selection_sha256': selection_hash,
        'complete': [], 'failures': [], 'requested': args.expected_candidates,
        'previous_failures': prior.get('previous_failures', []) + prior.get('failures', [])}
    jobs = [Path(p) for m in load(args.frontier / 'prepare.json')['models'] for p in m['jobs']]
    assert len(jobs) == args.expected_candidates
    def execute(path):
        job = load(path); model = Path(job['common']).parent.name
        output = args.output / model / (job['name'] + '.json')
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            value = load(output)
            assert value['status'] == 'complete' and value['job_sha256'] == digest(path)
            assert value['private_build_report_sha256'] == digest(job['report'])
            return {'model': model, 'name': job['name'], 'report_sha256': digest(output)}
        with output.with_suffix('.log').open('w') as log:
            process = subprocess.run([sys.executable, '-m', 'tools.audit_tree_node_rows',
                '--job', str(path), '--output', str(output)], stdout=log, stderr=subprocess.STDOUT)
        assert process.returncode == 0, (model, job['name'], process.returncode)
        return {'model': model, 'name': job['name'], 'report_sha256': digest(output)}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(execute, p): p for p in jobs}
        for future in as_completed(futures):
            try:
                report['complete'].append(future.result())
            except Exception as error:
                report['failures'].append({'job': str(futures[future]), 'error': repr(error)})
            write(args.output / 'report.json', report)
            if len(report['complete']) % 50 == 0:
                print(json.dumps({'reopened': len(report['complete']), 'failures': len(report['failures'])}), flush=True)
    assert digest(args.frontier / 'selection.json') == selection_hash
    report['status'] = 'complete' if len(report['complete']) == args.expected_candidates and not report['failures'] else 'failed'
    write(args.output / 'report.json', report)
    assert report['status'] == 'complete'


if __name__ == '__main__':
    main()
