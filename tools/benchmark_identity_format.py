"""Paired v2/v3 exports of verified trained rows; no training or GPU usage.

Keeps codec payloads, block sizes, original requests and reader settings matched.
Runs each reader in a fresh process, including the old library on v2 artifacts.
The prior fixture is read-only. Exported bundles also support complete-model
reopen, but this CPU benchmark measures row gathering, not model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .benchmark_counterfact_storage import environment, write
from .engram_row_adapter import file_sha256
from .native_row_store import HEADER, DESC, NativeRows, pack_native


def layout(path):
    """Hash descriptors without absolute offsets and hash the codec payload."""
    with path.open('rb') as f:
        magic, version, rows, width, unique, block_rows, count = HEADER.unpack(f.read(HEADER.size))
        if magic != b'MEFTNAT1' or version not in (2, 3):
            raise ValueError('v2/v3 table required')
        mapping_bytes = 0 if version == 3 else rows * 8
        f.seek(HEADER.size + mapping_bytes)
        descriptors = [DESC.unpack(f.read(DESC.size)) for _ in range(count)]
        payload = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            payload.update(chunk)
    return {'header': [rows, width, unique, block_rows, count],
            'descriptors_without_offsets_sha256': hashlib.sha256(
                b''.join(DESC.pack(0, *d[1:]) for d in descriptors)).hexdigest(),
            'payload_sha256': payload.hexdigest()}


def verify_rows(path, library, expected, rows):
    digest = hashlib.sha256()
    with NativeRows(path, library) as table:
        for start in range(0, rows, 4096):
            digest.update(table.rows(range(start, min(start + 4096, rows))))
    if digest.hexdigest() != expected:
        raise AssertionError('original-ID row reconstruction differs')
    return digest.hexdigest()


def prepare(fixture, output, library, old_library, blocks):
    m = json.loads((fixture / 'manifest.json').read_text())
    if m['status'] != 'passed' or file_sha256(fixture / 'rows.f32') != m['source_row_sha256']:
        raise ValueError('verified trained source rows required')
    if file_sha256(fixture / 'queries.npz') != m['queries_sha256']:
        raise ValueError('source query oracle changed')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(fixture / 'queries.npz', output / 'queries.npz')
    # Source remains read-only, and native workers do not open this link.
    (output / 'rows.f32').symlink_to((fixture / 'rows.f32').resolve())
    entry = {**m, 'status': 'building', 'environment': environment(), 'variants': [],
             'source_fixture_manifest_sha256': file_sha256(fixture / 'manifest.json'),
             'old_library_sha256': file_sha256(old_library), 'new_library_sha256': file_sha256(library),
             'paired_layouts': [],
             'diagnostic': 'Matched codec/block v2/v3 format ablation; no new training or GPU/model measurement'}
    original_meta = json.loads((fixture / 'xor/memory/embedding.json').read_text())
    recipe = json.loads((fixture / 'xor/model-recipe.json').read_text())
    for block in blocks:
        pair = []
        for implicit in (False, True):
            version = 3 if implicit else 2
            directory = output / f'v{version}-b{block}'
            memory = directory / 'memory'
            memory.mkdir(parents=True)
            started = time.perf_counter()
            packed = pack_native(fixture / 'rows.f32', memory / 'table.men', m['shape'][1] * 4,
                block, False, 'xor', implicit_identity=implicit)
            packed['export_seconds'] = time.perf_counter() - started
            packed['reconstructed_sha256'] = verify_rows(memory / 'table.men', library, m['source_row_sha256'], m['shape'][0])
            packed['layout'] = layout(memory / 'table.men')
            if not implicit:
                packed['old_reader_reconstructed_sha256'] = verify_rows(
                    memory / 'table.men', old_library, m['source_row_sha256'], m['shape'][0])
                if block == original_meta['block_rows']:
                    if file_sha256(memory / 'table.men') != file_sha256(fixture / 'xor/memory/table.men'):
                        raise AssertionError('legacy opt-out changed existing v2 bytes')
                    packed['matches_previous_v2_artifact'] = True
            else:
                try:
                    with NativeRows(memory / 'table.men', old_library):
                        raise AssertionError('old reader silently accepted v3')
                except ValueError as error:
                    packed['old_reader_rejection'] = str(error)
            meta = {**original_meta, 'block_rows': block, 'artifact_sha256': file_sha256(memory / 'table.men'),
                    'artifact_bytes': packed['artifact_bytes'], 'row_map_bytes': packed['row_map_bytes'],
                    'format_version': version}
            write(memory / 'embedding.json', meta)
            write(directory / 'model-recipe.json', {**recipe, 'memory_manifest_sha256': file_sha256(memory / 'embedding.json')})
            for filename in ('vocabulary.npz', 'nonrow-delta.safetensors'):
                (directory / filename).symlink_to((fixture / filename).resolve())
            entry['variants'].append({'name': directory.name, 'block_rows': block, 'packed': packed,
                'table_sha256': file_sha256(memory / 'table.men'),
                'memory_manifest_sha256': file_sha256(memory / 'embedding.json'),
                'memory_artifact_bytes': sum(p.stat().st_size for p in memory.iterdir()),
                'all_original_rows_byte_exact': True})
            pair.append(packed)
        if pair[0]['layout'] != pair[1]['layout']:
            raise AssertionError('codec bytes or physical layout changed')
        saved = pair[0]['artifact_bytes'] - pair[1]['artifact_bytes']
        if saved != m['shape'][0] * 8:
            raise AssertionError('format byte savings differ from map size')
        entry['paired_layouts'].append({'block_rows': block, 'saved_bytes': saved,
                                       'identical_codec_payload_and_descriptors': True})
        print(json.dumps({'fixture': fixture.name, 'block_rows': block, 'saved_bytes': saved}), flush=True)
    entry['status'] = 'passed'
    write(output / 'manifest.json', entry)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--old-library', type=Path, required=True)
    p.add_argument('--block-sizes', default='32,128,512,2048')
    p.add_argument('--cache-sizes', default='0,1048576')
    p.add_argument('--repeat', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
    args = p.parse_args()
    blocks = [int(n) for n in args.block_sizes.split(',')]
    caches = [int(n) for n in args.cache_sizes.split(',')]
    if (not blocks or min(blocks) < 1 or len(set(blocks)) != len(blocks) or
            not caches or min(caches) < 0 or len(set(caches)) != len(caches) or
            args.repeat < 1 or args.warmup < 0 or args.output.exists()):
        p.error('fresh output, distinct valid sizes and positive repeat required')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    prepare(args.fixture, args.output, args.native_library, args.old_library, blocks)
    schedule = []
    for i, block in enumerate(blocks):
        for j, cache in enumerate(caches):
            readers = [('old-v2', args.old_library, f'v2-b{block}'),
                       ('new-v2', args.native_library, f'v2-b{block}'),
                       ('new-v3', args.native_library, f'v3-b{block}')]
            # Alternate order across settings; each comparison is still one
            # process per reader, not independent repeated process trials.
            for label, lib, mode in readers if (i + j) % 2 == 0 else readers[::-1]:
                name = f'{label}-b{block}-c{cache}'
                command = [sys.executable, '-m', 'tools.benchmark_counterfact_storage', 'worker',
                    '--fixture', str(args.output), '--native-library', str(lib), '--mode', mode,
                    '--output', str(args.output / (name + '.json')), '--cache-bytes', str(cache),
                    '--repeat', str(args.repeat), '--warmup', str(args.warmup)]
                schedule.append({'name': name, 'command': command})
    write(args.output / 'schedule.json', schedule)
    for item in schedule:
        with (args.output / (item['name'] + '.log')).open('w') as log:
            subprocess.run(item['command'], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({'fixture': args.fixture.name, 'completed_worker': item['name']}), flush=True)
    write(args.output / 'complete.json', {'status': 'complete', 'workers': len(schedule)})


if __name__ == '__main__':
    main()
