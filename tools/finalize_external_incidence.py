"""Validate the complete declared build grid and derive source-linked figures."""
import argparse
import json
from pathlib import Path

from .external_cotraining import file_hash


def validate_grid(report, preparation, verifier):
    expected = {(250000, 'legacy', None), (250000, 'external', 64),
                (1000000, 'legacy', None), (1000000, 'external', 64),
                (1855907, 'legacy', None), (1855907, 'external', 64),
                (1855907, 'external', 128), (1855907, 'external', 256)}
    if report.get('status') != 'completed' or report.get('failures') or \
            preparation.get('status') != 'completed' or verifier.get('status') != 'completed':
        raise ValueError('incomplete or failed evidence; do not publish a success summary')
    keys = [(r['input']['rows'], r['job']['method'], r['job']['memory_mib']) for r in report['runs']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('missing, duplicate or unexpected declared build')
    prepared = {s['rows']: s for s in preparation['sets']}
    if len(preparation['sets']) != 3 or set(prepared) != {250000, 1000000, 1855907}:
        raise ValueError('preparation scales differ from the declared grid')
    bases = {r['input']['rows']: r for r in report['runs'] if r['job']['method'] == 'legacy'}
    rows = []
    for r in report['runs']:
        n = r['input']['rows']
        base = bases[n]
        if r['source_commit'] != report['source_commit'] or r['input'] != prepared[n] or \
                r['partition'] != base['partition'] or \
                r['source_files_sha256'] != base['source_files_sha256']:
            raise ValueError('mixed source, input or partition')
        if r['status'] != 'completed' or r['device'] != 'cpu' or \
                r['projection']['source_sha256'] != r['input']['source_sha256'] or \
                r['projection']['means_sha256'] != base['projection']['means_sha256']:
            raise ValueError('unfinished build or changed source/means')
        external = r['job']['method'] == 'external'
        if external:
            expected_bytes = {'mapping': n * 8, 'sizes': r['partition']['group_rows'] * 8,
                              'means': r['partition']['group_rows'] * r['input']['dim'] * 4}
            if r.get('all_output_bytes_identical') != expected_bytes:
                raise ValueError('missing full byte comparisons')
        rows.append({'rows': n, 'method': r['job']['method'], 'managed_memory_mib': r['job']['memory_mib'],
            'groups': r['partition']['group_rows'], 'peak_rss_mib': r['peak_rss_bytes'] / 2**20,
            'total_seconds': r['total_seconds'], 'partition_seconds': r['partition_seconds'],
            'projection_seconds': r['projection_seconds'],
            'sampled_workspace_mib': r['sampled_peak_workspace_logical_bytes'] / 2**20,
            'final_artifact_mib': r['final_artifact_bytes'] / 2**20,
            'temporary_accumulator_mib': r['projection'].get('temporary_accumulator_bytes', 0) / 2**20,
            'rss_reduction_fraction': 1 - r['peak_rss_bytes'] / base['peak_rss_bytes'],
            'rss_reduction_factor': base['peak_rss_bytes'] / r['peak_rss_bytes'],
            'build_time_factor': r['total_seconds'] / base['total_seconds'],
            'all_output_bytes_identical': r.get('all_output_bytes_identical')})
    full = bases[1855907]['partition']
    full_run = bases[1855907]
    if verifier.get('means_sha256') != full_run['projection']['means_sha256'] or any(
            verifier['source_files_sha256'].get(name) != full_run['source_files_sha256'][name]
            for name in ('external_cotraining.py', 'cotraining_groups.py')):
        raise ValueError('native verifier used different means or construction code')
    if verifier['partition'] != full or not verifier.get('partition_arrays_and_legacy_reader_identical') or \
            verifier.get('all_group_means_identical_to_previous_native_rows') != full['group_rows'] or \
            verifier['plan']['bags'] != 512 or verifier['plan']['partition_sha256'] != full['mapping_sha256'] or \
            verifier.get('query_bags_sha256') != 'cedaa2f0ac19ceedfea4eda07139204d8c179d84e3f33e213811168cd93f83fe' or \
            set(verifier.get('identical_plan_arrays', {})) != {'ids', 'weights', 'offsets', 'denominators'} or \
            set(verifier.get('identical_native_outputs', {})) != {'sum', 'mean'}:
        raise ValueError('missing independent full native verification')
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = {'legacy': '#a04e35', 'external': '#146f8a'}
    for method in ('legacy', 'external'):
        selected = [r for r in rows if r['method'] == method and
                    (method == 'legacy' or r['managed_memory_mib'] == 64)]
        x = [r['rows'] / 1e6 for r in selected]
        for ax, key, label in ((axes[0, 0], 'peak_rss_mib', 'Peak process RSS (MiB)'),
                                (axes[0, 1], 'total_seconds', 'Complete build time (seconds)')):
            ax.plot(x, [r[key] for r in selected], 'o-', color=colors[method],
                    label='In-memory baseline' if method == 'legacy' else 'External: 64 MiB budget')
            ax.set_xlabel('Actual trained rows (millions)')
            ax.set_ylabel(label)
            ax.legend(fontsize=9)
    full = [r for r in rows if r['rows'] == 1855907]
    labels = ['In-memory', 'Ext 64', 'Ext 128', 'Ext 256']
    axes[1, 0].bar(labels, [r['peak_rss_mib'] for r in full],
                   color=[colors[r['method']] for r in full])
    axes[1, 0].set_ylabel('Peak process RSS (MiB)')
    for i, r in enumerate(full):
        axes[1, 0].text(i, r['peak_rss_mib'] + 60, f"{r['peak_rss_mib']:.1f}", ha='center', fontsize=9)
    axes[1, 0].set_ylim(0, max(r['peak_rss_mib'] for r in full) * 1.15)
    axes[1, 1].bar(labels, [r['sampled_workspace_mib'] for r in full], color='#146f8a',
                   label='Sampled logical workspace peak')
    axes[1, 1].bar(labels, [r['final_artifact_mib'] for r in full], color='#93b5b8',
                   label='Final build outputs')
    axes[1, 1].set_ylabel('Logical disk (MiB)')
    axes[1, 1].legend(fontsize=9)
    titles = ['RAM at the fixed 64 MiB managed budget', 'Lower RAM costs more build time',
              'Full table: three managed budgets', 'Full table: disk is part of the tradeoff']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('NGramStore: external incidence construction on existing trained rows', fontsize=14)
    fig.supxlabel('CPU only; shared warm host. Budget is not an RSS/page-cache cap. No new training or LLM quality claim.',
                  fontsize=9)
    fig.savefig(output / 'construction.png', dpi=160)
    fig.savefig(output / 'construction.svg')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    cfg = p.parse_args()
    paths = {k: cfg.evidence / filename for k, filename in
             (('report', 'report.json'), ('preparation', 'preparation.json'), ('verifier', 'native-verifier.json'))}
    data = {k: json.loads(path.read_text()) for k, path in paths.items()}
    if data['report']['preparation_report_sha256'] != file_hash(paths['preparation']):
        raise ValueError('preparation provenance changed')
    rows = validate_grid(**data)
    summary = {'source_reports_sha256': {k: file_hash(path) for k, path in paths.items()},
               'runs': rows, 'preparation_seconds': data['preparation']['seconds'],
               'preparation_peak_rss_mib': data['preparation']['peak_rss_bytes'] / 2**20,
               'resident_verifier_peak_rss_mib': data['verifier']['peak_rss_bytes'] / 2**20,
               'limitations': data['report']['limitations']}
    (cfg.evidence / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(rows, cfg.evidence)
    files = sorted(p for p in cfg.evidence.rglob('*') if p.is_file() and p.name != 'manifest.json')
    manifest = {str(p.relative_to(cfg.evidence)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in files}
    (cfg.evidence / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
