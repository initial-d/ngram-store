"""Registered independent-row byte packing over existing trained Qwen codes.

Uses real calibration/test original-token requests and the unchanged codec
codebooks. This is a storage experiment, never a new quantizer or trained model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .tree_node_quality import digest, load, serving_files, write

CODECS = ('fp32', 'sq4', 'sq6', 'sq8', 'pq-8x8', 'pq-16x8')


def prepare_model(args, prepared_path):
    from .faiss_row_codec import pack_codec_native
    cfg = load(prepared_path); output = args.output / prepared_path.parent.name
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for codec in CODECS:
        direct = next(e for e in cfg['entries'] if e['codec'] == codec and e['variant'] == 'direct')
        job = load(direct['job']); original_report = load(job['report'])
        direct_path = output / 'jobs' / ('direct-' + codec + '.json')
        write(direct_path, {**job, 'native_library': str(args.library)})
        entries.append({'codec': codec, 'variant': 'direct', 'job': str(direct_path)})
        for block_rows in (16, 64, 256, 1024):
            layouts = [('row-xor', 'none'), *[(c, p) for c in ('zlib', 'lz4', 'zstd') for p in ('none', 'xor')]]
            for block_codec, predictor in layouts:
                name = f'{codec}-{block_codec}-rows{block_rows}-{predictor}'
                directory = output / 'bundles' / name
                if not (directory / 'codec.json').exists():
                    pack_codec_native(job['directory'], directory, native_library=args.library,
                        max_block_bytes=None, block_rows=block_rows, predictor=predictor,
                        block_codec=block_codec)
                metadata = load(directory / 'codec.json')
                files = serving_files(directory)
                report = {**original_report, 'name': name, 'serving_files': files,
                    'row_bundle_bytes': sum(f['bytes'] for f in files.values()),
                    'original_code_bytes_sha256': metadata['codes_storage']['original_code_bytes_sha256'],
                    'layout_source_directory': job['directory'],
                    'layout_source_metadata_sha256': digest(Path(job['directory']) / 'codec.json')}
                report['model_bytes'] = report['row_bundle_bytes'] + report['common_model_bytes']
                report_path = output / 'reports' / (name + '.json'); write(report_path, report)
                path = output / 'jobs' / (name + '.json')
                write(path, {**job, 'name': name, 'directory': str(directory), 'report': str(report_path),
                    'native_library': str(args.library)})
                entries.append({'codec': codec, 'variant': 'row-packets' if block_codec == 'row-xor' else block_codec,
                    'predictor': predictor, 'block_rows': block_rows, 'job': str(path)})
    result = output / 'prepared.json'
    write(result, {'status': 'complete', 'entries': entries, 'source_prepared_sha256': digest(prepared_path),
        'source_sha256': digest(__file__), 'protocol_sha256': digest(args.protocol),
        'library': str(args.library), 'library_sha256': digest(args.library)})
    return str(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='phase', required=True)
    prepare = sub.add_parser('prepare')
    for name in ('sources', 'output', 'library', 'protocol'):
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--workers', type=int, default=6)
    run = sub.add_parser('lane')
    for name in ('prepared', 'output', 'upstream'):
        run.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.phase == 'prepare':
        paths = sorted(args.sources.glob('*/prepared.json'))
        if len(paths) != 6: raise ValueError('complete six existing Qwen sources required')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            models = list(pool.map(lambda path: prepare_model(args, path), paths))
        write(args.output / 'prepared.json', {'status': 'complete', 'models': models,
            'protocol_sha256': digest(args.protocol)})
    else:
        from .controlled_native_deployment import lane, select_layouts
        args.cpu_only = True; args.requests = 32
        args.selection_variant = 'row-packets'; args.minimum_saving = .01
        lane(args)
        completed = load(args.output / 'cpu-progress.json')['completed']
        choices = {}
        for variant in ('row-packets', 'zlib', 'lz4', 'zstd'):
            for saving in (.01, .10):
                choices[f'{variant}-{saving:g}'] = select_layouts(completed, variant, saving)
        write(args.output / 'capacity-selection.json', {'status': 'frozen', 'test_used': False,
            'source_sha256': digest(__file__), 'choices': choices,
            'rule': 'For each unchanged decoded codec and byte compressor, minimize calibration complete-token p50 subject to 1% or 10% complete-branch savings; direct fallback. Official test confirms all frozen selections.',
            'cpu_reports_sha256': {e['report']: digest(e['report']) for e in completed}})


if __name__ == '__main__':
    main()
