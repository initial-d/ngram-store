"""Analyze same-task trained-row FAISS comparisons without choosing on test."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def block_interval(a, b, positions, repeats=1000, seed=42):
    groups, inverse = np.unique(positions // 128, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    totals = {prefix + key: np.bincount(inverse, weights=scores[key])
              for prefix, scores in (('a', a), ('b', b)) for key in ('loss', 'discrimination')}
    rng = np.random.default_rng(seed)
    intervals = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(groups), len(groups))
        intervals.append([totals['aloss'][chosen].sum() / totals['bloss'][chosen].sum() - 1,
                          (totals['adiscrimination'][chosen].sum() - totals['bdiscrimination'][chosen].sum()) / count[chosen].sum()])
    return {'token_block_size': 128, 'observed_blocks': len(groups), 'repeats': repeats, 'seed': seed,
            'loss_relative_change_95_percent_interval': np.quantile(intervals, [.025, .975], axis=0)[:, 0].tolist(),
            'discrimination_change_95_percent_interval': np.quantile(intervals, [.025, .975], axis=0)[:, 1].tolist(),
            'scope': 'Paired within-text token-block bootstrap; exploratory intervals, not independent corpora or simultaneous bounds.'}


def merge_reports(base, supplement, dedup=None):
    """Merge a separately disclosed rate extension; reselect using validation."""
    base = json.loads(json.dumps(base))
    extra = {(r['order'], r['seed']): r for r in supplement['runs']}
    for run in base['runs']:
        other = extra.pop((run['order'], run['seed']))
        for key in ('checkpoint_sha256', 'partition', 'training_original_ids_sha256', 'evaluation', 'traces'):
            if run[key] != other[key]:
                raise ValueError('supplement changes matched input: ' + key)
        if set(v['name'] for v in run['variants']) & set(v['name'] for v in other['variants']):
            raise ValueError('duplicate codec variants')
        run['phase_selections'] = [run['selection'], other['selection']]
        run['variants'].extend(other['variants'])
        run['failures'].extend(other['failures'])
        if dedup is not None:
            controls = [r for r in dedup['runs'] if r['order'] == run['order'] and r['seed'] == run['seed']]
            for control in controls:
                if control['checkpoint_sha256'] != run['checkpoint_sha256']:
                    raise ValueError('dedup checkpoint mismatch')
                if control['status'] == 'completed':
                    run['variants'].append(control['variant'])
                elif control['status'] == 'failed':
                    run['failures'].append({'name': control['name'], 'error': 'dedup worker failed'})
        settings = base['settings']
        for budget, choices in run['selection'].items():
            for family in choices:
                eligible = [v for v in run['variants'] if
                    (v['name'].startswith(('direct-', 'dedup-')) if family == 'direct' else
                     v['name'].endswith('-hybrid') if family == 'hybrid' else
                     v['name'].startswith('group-') and not v['name'].endswith('-hybrid'))
                    and v['quality']['validation']['loss_relative_change'] <= float(budget)
                    and v['quality']['validation']['discrimination_change'] >= -settings['max_discrimination_drop']
                    and v['known_word_aggregate_fidelity']['global_relative_l2'] <= settings['max_aggregate_error']]
                chosen = min(eligible, key=lambda v: v['model_bytes']) if eligible else None
                choices[family] = None if chosen is None else {'name': chosen['name'], 'model_bytes': chosen['model_bytes'],
                    'test': chosen['quality']['test'], 'test_quality_budget_passed':
                    chosen['quality']['test']['loss_relative_change'] <= float(budget)
                    and chosen['quality']['test']['discrimination_change'] >= -settings['max_discrimination_drop']}
    if extra:
        raise ValueError('supplement has unmatched checkpoints')
    base['supplemental_grid_protocol'] = {k: v for k, v in supplement.items() if k != 'runs'}
    if dedup is not None:
        base['encoded_byte_dedup'] = dedup
    base['status'] = 'completed' if base['status'] == supplement['status'] == 'completed' else 'completed-with-failures'
    return base


def analyze(report, root=None, supplement_root=None, dedup_root=None):
    rows, selections = [], []
    for run in report['runs']:
        by_name = {v['name']: v for v in run['variants']}
        for v in run['variants']:
            codec = v['codec']['codec']
            family = 'hybrid' if v['name'].endswith('-hybrid') else 'group' if v['name'].startswith('group-') else 'dedup' if v['name'].startswith('dedup-') else 'direct'
            row = {'order': run['order'], 'seed': run['seed'], 'name': v['name'], 'family': family,
                   'codec': codec, 'model_bytes': v['model_bytes'],
                   'model_plus_fixed_plans_bytes': v['model_bytes'] + v['fixed_trace_plan_bytes'],
                   'known_aggregate_relative_l2': v['known_word_aggregate_fidelity']['global_relative_l2'],
                   'known_p50_ms': v['query']['known']['p50_ms'],
                   'known_dynamic_p50_ms': v['query']['known']['with_schedule_compilation']['p50_ms'],
                   'known_p95_ms': v['query']['known']['p95_ms'],
                   'oov_p50_ms': v['query']['oov']['p50_ms'],
                   'oov_accepted': v['query']['oov']['fidelity']['accepted'],
                   'oov_total': v['query']['oov']['fidelity']['total'],
                   'query_peak_rss_bytes': v['query_peak_rss_bytes'],
                   'training_seconds': v['codec']['training_seconds'],
                   'encoding_seconds': v['codec']['encoding_seconds'],
                   'validation': v['quality']['validation'], 'test': v['quality']['test']}
            direct = by_name.get('direct-' + codec)
            if direct and family != 'direct':
                row['versus_matched_direct'] = {
                    'model_saving_fraction': 1 - v['model_bytes'] / direct['model_bytes'],
                    'known_replay_speedup': direct['query']['known']['p50_ms'] / v['query']['known']['p50_ms'],
                    'known_dynamic_speedup': direct['query']['known']['with_schedule_compilation']['p50_ms'] / v['query']['known']['with_schedule_compilation']['p50_ms'],
                    'aggregate_error_ratio': row['known_aggregate_relative_l2'] / max(direct['known_word_aggregate_fidelity']['global_relative_l2'], 1e-30)}
            rows.append(row)
        for budget, selected in run['selection'].items():
            for family, item in selected.items():
                row = {'order': run['order'], 'seed': run['seed'], 'loss_budget': float(budget),
                       'family': family, 'selected': item}
                if item is not None and root is not None:
                    directory = root / f"order-{run['order']}-seed-{run['seed']}"
                    if not (directory / (item['name'] + '-test-scores.npz')).exists() and supplement_root is not None:
                        directory = supplement_root / directory.name
                    if not (directory / (item['name'] + '-test-scores.npz')).exists() and dedup_root is not None:
                        directory = dedup_root / directory.name
                    candidate = dict(np.load(directory / (item['name'] + '-test-scores.npz')))
                    common = root / directory.name / 'common'
                    baseline = dict(np.load(common / 'test-scores.npz'))
                    positions = np.load(common / 'test-evaluation.npz')['positions']
                    row['test_paired_block_interval'] = block_interval(candidate, baseline, positions)
                selections.append(row)
    return {'source_commit': report['source_commit'], 'status': report['status'],
            'runs': len(report['runs']), 'variants': len(rows),
            'failed_runs': [{ 'order': r['order'], 'seed': r['seed'], 'failures': r['failures']}
                            for r in report['runs'] if r['failures']],
            'rows': rows, 'validation_selected': selections,
            'interpretation': ['Known-word closed-domain comparison permits direct and certified-only groups on the SAME bags.',
                'OOV/all-domain comparisons must use direct or hybrid; rejected work is never counted as a speedup.',
                'Reconstruction error, finite SGNS quality and query cost are separate outcomes.',
                'Codebook/transform overhead is material on small tables; not extrapolated to million-row scale.',
                'Matched-codec grouping savings are distinct from choosing a different codec on validation.']}


def precomputed_worker(job):
    job = json.loads(Path(job).read_text())
    table = np.load(job['table'], mmap_mode='r')
    ids = np.asarray(job['ids'], dtype=np.int64)
    for _ in range(2):
        table[ids]
    samples = []
    for _ in range(job['repeat']):
        start = time.perf_counter()
        values = table[ids]
        samples.append(time.perf_counter() - start)
    np.testing.assert_array_equal(values, np.load(job['reference']))
    return {'scope': 'Known-word inference only; no arbitrary n-gram query, OOV, or matched training.',
            'model_bytes': job['model_bytes'], 'p50_ms': float(np.median(samples) * 1000),
            'p95_ms': float(np.quantile(samples, .95) * 1000), 'samples_seconds': samples,
            'aggregate_relative_l2': 0., 'known_bags': len(ids), 'oov_accepted': 0}


def serving_controls(report, root, output):
    from .cotraining_groups import BagPartition
    controls = []
    for run in report['runs']:
        name = f"order-{run['order']}-seed-{run['seed']}"
        common = root / name / 'common'
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        original = BagPartition.load(common / 'domain.npz')
        dtype = next(t for t in (np.uint8, np.uint16, np.uint32, np.uint64)
                     if len(original.sizes) - 1 <= np.iinfo(t).max)
        compact = directory / 'compact-domain.npz'
        np.savez(compact, mapping=original.mapping.astype(dtype), domain_sha256=original.domain_sha256,
                 domain_bags=original.domain_bags)
        decoded = BagPartition.load(compact)
        if decoded.summary() != original.summary():
            raise ValueError('compact certificate changes mapping/domain')
        words = json.loads((common / 'registry.json').read_text())[0]
        word_registry = directory / 'word-registry.json'
        word_registry.write_text(json.dumps(words, ensure_ascii=True))
        ids = json.loads((common / 'trace-bags.json').read_text())['known'][1]
        oov_bags = json.loads((common / 'trace-bags.json').read_text())['oov'][0]
        job = {'table': str(common / 'all-reference.npy'), 'ids': ids,
               'reference': str(common / 'known-reference.npy'), 'repeat': report['settings']['repeat'],
               'model_bytes': (common / 'all-reference.npy').stat().st_size
                   + (common / 'output-rows.npy').stat().st_size + word_registry.stat().st_size}
        job_file = directory / 'precomputed-job.json'
        job_file.write_text(json.dumps(job))
        process = subprocess.run([sys.executable, '-m', 'tools.analyze_faiss_cotraining',
                                  '--precomputed-job', str(job_file)], capture_output=True, text=True, check=True)
        controls.append({'order': run['order'], 'seed': run['seed'],
            'oov_nonempty_bags': sum(bool(bag) for bag in oov_bags),
            'certified_nonempty_oov_bags': sum(bool(bag) and original.certifies(bag) for bag in oov_bags),
            'compact_domain': {'original_bytes': (common / 'domain.npz').stat().st_size,
                'compact_bytes': compact.stat().st_size, 'dtype': np.dtype(dtype).name,
                'mapping_sha256': decoded.summary()['mapping_sha256'],
                'scope': 'Separate on-disk representation ablation; query timings use the original uint64 certificate. Runtime map remains int64.'},
            'precomputed_known_words': json.loads(process.stdout)})
    return controls


def figures(analysis, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'direct': ('o', '#356bb3'), 'group': ('^', '#db7e1a'), 'hybrid': ('x', '#ad4663'), 'dedup': ('s', '#478653')}
    for metric, filename, ylabel in (
            ('known_aggregate_relative_l2', 'capacity-error.svg', 'Known-word aggregate relative L2'),
            ('known_p50_ms', 'capacity-replay.svg', '512-bag replay p50 (ms)'),
            ('known_dynamic_p50_ms', 'capacity-dynamic.svg', 'ID schedule + query p50 (ms)')):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, order in zip(axes, (4, 8, 12)):
            for family, (marker, color) in styles.items():
                for codec in sorted({r['codec'] for r in analysis['rows']}):
                    values = [r for r in analysis['rows'] if r['order'] == order and r['family'] == family and r['codec'] == codec]
                    if not values:
                        continue
                    x = np.median([v['model_bytes'] / 2**20 for v in values])
                    y = max(1e-9, np.median([v[metric] for v in values]))
                    ax.scatter(x, y, marker=marker, color=color, s=45)
                    if family != 'hybrid' and codec in ('fp32', 'sq8', 'sq6', 'sq4', 'pq-64x8', 'opq-64x8'):
                        ax.annotate(codec, (x, y), fontsize=6, xytext=(3, 3), textcoords='offset points', color=color)
                ax.scatter([], [], marker=marker, color=color, label=family)
            ax.set_title(f'Max n-gram order {order}')
            ax.set_xlabel('Complete model (MiB)')
            ax.set_yscale('log')
            ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.suptitle('Finite SGNS / known-word domain; medians over 3 training seeds', fontsize=11)
        fig.text(.5, .01, 'Word/output rows, registry, codebooks and maps counted; hybrid includes full original-row fallback. Zero error plotted at 1e-9.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .94))
        fig.savefig(output / filename)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path)
    p.add_argument('--supplement-report', type=Path)
    p.add_argument('--dedup-report', type=Path)
    p.add_argument('--raw-root', type=Path)
    p.add_argument('--supplement-root', type=Path)
    p.add_argument('--dedup-root', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--precomputed-job', type=Path)
    p.add_argument('--plots', action='store_true')
    cfg = p.parse_args()
    if cfg.precomputed_job:
        print(json.dumps(precomputed_worker(cfg.precomputed_job)))
        return
    if cfg.report is None or cfg.output_dir is None:
        p.error('report and output-dir required')
    raw = gzip.decompress(cfg.report.read_bytes()) if cfg.report.suffix == '.gz' else cfg.report.read_bytes()
    report = json.loads(raw)
    if cfg.supplement_report:
        supplement_raw = gzip.decompress(cfg.supplement_report.read_bytes()) if cfg.supplement_report.suffix == '.gz' else cfg.supplement_report.read_bytes()
        report = merge_reports(report, json.loads(supplement_raw), json.loads(cfg.dedup_report.read_text()) if cfg.dedup_report else None)
    analysis = analyze(report, cfg.raw_root, cfg.supplement_root, cfg.dedup_root)
    analysis['report_sha256'] = hashlib.sha256(raw).hexdigest()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.raw_root:
        analysis['serving_controls'] = serving_controls(report, cfg.raw_root, cfg.output_dir / 'controls')
    if cfg.supplement_report:
        (cfg.output_dir / 'merged-report.json.gz').write_bytes(gzip.compress(json.dumps(report, indent=2).encode(), mtime=0))
    (cfg.output_dir / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
    if cfg.plots:
        figures(analysis, cfg.output_dir)
    print(json.dumps({'runs': analysis['runs'], 'variants': analysis['variants'], 'failed_runs': analysis['failed_runs']}))


if __name__ == '__main__':
    main()
