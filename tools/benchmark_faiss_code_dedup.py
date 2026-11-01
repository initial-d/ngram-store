"""Strong inference control: deduplicate IDENTICAL actual FAISS row codes.

Reuses trained codebooks without refitting. All original IDs and query bags
remain supported at the original codec's approximation; no incidence certificate
or FP32 fallback is needed. Negative complete-size results are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .benchmark_faiss_cotraining import write_json, schedule, files_size
from .faiss_row_codec import FaissRows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    cfg = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('CPU-only: empty CUDA_VISIBLE_DEVICES required')
    cfg.output_dir = cfg.output_dir.resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'scope': 'Identical trained-code dedup, full-domain frozen inference; no shared-training contract.',
              'runs': []}
    jobs = []
    for root in cfg.roots:
        source_report = json.loads((root / 'report.json').read_text())
        for run in source_report['runs']:
            run_name = f"order-{run['order']}-seed-{run['seed']}"
            common = root / run_name / 'common'
            out = cfg.output_dir / run_name
            out.mkdir(exist_ok=True)
            traces = json.loads((common / 'trace-bags.json').read_text())
            word_bags = dict(np.load(root / run_name / 'direct-fp32-plans/all.npz')) if (root / run_name / 'direct-fp32-plans/all.npz').exists() else None
            if word_bags is None:
                # Rate extension has no FP32 candidate: ANY direct plan has the
                # same original-ID bag schedule before code-byte compaction.
                path = next((root / run_name).glob('direct-*-plans/all.npz'))
                word_bags = dict(np.load(path))
            all_bags = [np.repeat(word_bags['ids'][lo:hi], word_bags['weights'][lo:hi].astype(np.int64)).tolist()
                        for lo, hi in zip(word_bags['offsets'][:-1], word_bags['offsets'][1:])]
            for direct in [v for v in run['variants'] if v['name'].startswith('direct-')]:
                name = 'dedup-' + direct['codec']['codec']
                codec_dir = out / name
                codec_dir.mkdir()
                original_dir = root / run_name / direct['name']
                original = FaissRows(original_dir)
                encoded = np.asarray(original.codes).view(np.uint8).reshape(len(original.codes), -1)
                unique_bytes, mapping = np.unique(encoded, axis=0, return_inverse=True)
                unique = unique_bytes.view(original.codes.dtype).reshape(-1, original.codes.shape[1])
                dtype = next(t for t in (np.uint8, np.uint16, np.uint32, np.uint64) if len(unique) - 1 <= np.iinfo(t).max)
                np.save(codec_dir / 'codes.npy', unique)
                np.save(codec_dir / 'original-to-code.npy', mapping.astype(dtype))
                metadata = json.loads((original_dir / 'codec.json').read_text())
                metadata['rows'] = len(unique)
                write_json(codec_dir / 'codec.json', metadata)
                if original.index is not None:
                    shutil.copyfile(original_dir / 'codec.faiss', codec_dir / 'codec.faiss')
                decoded = FaissRows(codec_dir)
                maximum = 0.
                for begin in range(0, len(mapping), 4096):
                    ids = np.arange(begin, min(begin + 4096, len(mapping)))
                    a, b = original.rows(ids), decoded.rows(mapping[ids])
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
                    maximum = max(maximum, float(np.abs(a - b).max()))
                model_bytes = run['common_word_output_registry_bytes'] + files_size(codec_dir)
                audit = {'order': run['order'], 'seed': run['seed'], 'name': name,
                    'checkpoint_sha256': run['checkpoint_sha256'], 'original_stored_rows': len(mapping),
                    'unique_code_rows': len(unique), 'mapping_dtype': np.dtype(dtype).name,
                    'all_original_code_rows_verified': len(mapping), 'maximum_codec_decode_difference': maximum,
                    'model_bytes': model_bytes, 'direct_model_bytes': direct['model_bytes'],
                    'complete_saving_fraction': 1 - model_bytes / direct['model_bytes'],
                    'code_rows_sha256': hashlib.sha256(original.codes.tobytes()).hexdigest()}
                report['runs'].append(audit)
                if model_bytes >= direct['model_bytes']:
                    audit['status'] = 'flat-preferred'
                    write_json(cfg.output_dir / 'report.json', report)
                    continue
                plan_dir = out / (name + '-plans')
                plan_dir.mkdir()
                plans = {}
                for trace_name, (bags, word_ids) in {**traces, 'all': (all_bags, word_bags['words'])}.items():
                    plan = schedule(bags, word_ids, code_mapping=mapping)
                    file = plan_dir / (trace_name + '.npz')
                    np.savez(file, **plan)
                    if trace_name != 'all':
                        plans[trace_name] = str(file)
                job = {'name': name, 'codec_dir': str(codec_dir), 'code_mapping': str(codec_dir / 'original-to-code.npy'),
                    'common': str(common), 'fallback': False, 'grouped': False, 'coverage': 'full-domain-code-dedup',
                    'model_bytes': model_bytes, 'fixed_trace_plan_bytes': sum(Path(f).stat().st_size for f in plans.values()),
                    'occurrences': {key: sum(map(len, bags)) for key, (bags, w) in traces.items()},
                    'trace_plans': plans, 'all_plan': str(plan_dir / 'all.npz'),
                    'repeat': source_report['settings']['repeat'], 'batch_bags': source_report['settings']['batch_bags'],
                    'report': str(out / (name + '-report.json'))}
                job_file = out / (name + '-job.json')
                write_json(job_file, job)
                jobs.append((job_file, audit))
                write_json(cfg.output_dir / 'report.json', report)
    for job_file, audit in np.random.default_rng(42).permutation(np.asarray(jobs, dtype=object)):
        result = subprocess.run([sys.executable, '-m', 'tools.benchmark_faiss_cotraining', '--worker', str(job_file)], capture_output=True, text=True)
        (job_file.with_suffix('.log')).write_text(result.stdout + result.stderr)
        audit['status'] = 'completed' if result.returncode == 0 else 'failed'
        if result.returncode == 0:
            audit['variant'] = json.loads(Path(json.loads(job_file.read_text())['report']).read_text())
        write_json(cfg.output_dir / 'report.json', report)
    report['status'] = 'completed' if all(r['status'] != 'failed' for r in report['runs']) else 'completed-with-failures'
    write_json(cfg.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
