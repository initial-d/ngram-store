"""Reproduce full-table CPU half-bag measurements in the existing shared workspace.

Frozen protocol: two full seed-3 trained classifier tables, AG News zero and
DBpedia uniform; existing FAISS FP16 bytes, plus explicitly RNE-rounded BF16
exports from their frozen FP32 rows. Raw identity layout, chunk 4096, original
256-document test traces, batch 1/32, 2 warmups, 10 repeats, 3 paired rounds.
Both baseline/candidate compile at O3 with the same portable codec flags.
Additional existing compressed AG News FP16 control uses batch 32, 3 repeats,
2 rounds. No training, GPU usage, random vectors or task-quality claims.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

WORK = Path('/llm-align/duyimin/ngram-half-bags-20261117')
ROOT = Path('/llm-align/duyimin/ngram-controlled-deployment-20261111')
PRIOR = Path('/llm-align/duyimin/ngram-sparse-xor-20261115')
MODELS = ['ag_news-exact-zero-s3-lr10', 'dbpedia_14-exact-uniform-s3-lr3']
COMMIT = '5f22b912cbab97081099ac08a2de15b1456122db'
sys.path.insert(0, str(WORK / 'runtime'))
from tools.native_row_store import HEADER, DESC  # noqa: E402


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            result.update(data)
    return result.hexdigest()


def build_table(model, dtype):
    begin = time.monotonic()
    directory = WORK / 'tables' / model / dtype
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {'model': model, 'stored_dtype': dtype, 'block_rows': 256,
        'layout': 'uncompressed implicit identity, no dedup, no row reordering',
        'scope': 'Frozen trained classifier rows; BF16 is a new rounded export, not BF16 training.'}
    if dtype == 'float16':
        codes_path = ROOT / 'application-frontier/models' / model / 'bundles/faiss-sq-fp16/codes.npy'
        codec_path = codes_path.with_name('codec.json')
        metadata.update(source=str(codes_path), codec=json.loads(codec_path.read_text()),
            codec_sha256=digest(codec_path), conversion='None: existing packed FAISS FP16 codes reused byte-exactly')
        codes = np.load(codes_path, mmap_mode='r', allow_pickle=False)
        if codes.ndim != 2 or codes.dtype != np.dtype('uint8') or codes.shape[1] % 2:
            raise ValueError('packed FAISS FP16 codes required')
    else:
        source_path = ROOT / 'training-registered' / model / 'rows.npy'
        source = np.load(source_path, mmap_mode='r', allow_pickle=False)
        if source.ndim != 2 or source.dtype != np.dtype('<f4'):
            raise ValueError('frozen little-endian FP32 source required')
        codes_path = directory / 'codes.npy'
        codes = np.lib.format.open_memmap(codes_path, mode='w+', dtype='<u2', shape=source.shape)
        for start in range(0, len(source), 65536):
            values = np.ascontiguousarray(source[start:start + 65536])
            if not np.isfinite(values).all():
                raise ValueError('finite source required for this rounded BF16 export')
            bits = values.view(np.uint32)
            codes[start:start + len(values)] = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        codes.flush()
        metadata.update(source=str(source_path), source_sha256=digest(source_path),
            conversion='FP32 -> BF16 round-to-nearest ties-to-even once before timing; changes source vectors, no further error in candidate')
    width = codes.shape[1] * codes.dtype.itemsize
    rows = len(codes); blocks = (rows + 255) // 256
    table = directory / 'rows.meft'
    # Write the already frozen contiguous NPY payload directly to version 3;
    # no temporary expanded table, random row generation or custom codec.
    position = HEADER.size + blocks * DESC.size
    with table.open('wb') as dest:
        dest.write(HEADER.pack(b'MEFTNAT1', 3, rows, width, rows, 256, blocks))
        for block in range(blocks):
            size = min(256, rows - block * 256) * width
            dest.write(DESC.pack(position, size, size, 0)); position += size
        with codes_path.open('rb') as src:
            src.seek(codes.offset)
            remaining = rows * width
            while remaining:
                data = src.read(min(8 << 20, remaining))
                if not data:
                    raise ValueError('truncated frozen codes')
                dest.write(data); remaining -= len(data)
        dest.flush(); os.fsync(dest.fileno())
    metadata.update(rows=rows, dim=width // 2, row_bytes=width, codes=str(codes_path),
        codes_sha256=digest(codes_path), table=str(table), table_sha256=digest(table),
        table_bytes=table.stat().st_size, build_seconds=time.monotonic() - begin)
    (directory / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('built', model, dtype, rows, flush=True)
    return metadata


def main():
    (WORK / 'results').mkdir(exist_ok=True)
    (WORK / 'lib').mkdir(exist_ok=True)
    flags = ['-I' + str(PRIOR / 'include'), '-DMEFT_USE_LZ4', '-DMEFT_USE_ZSTD',
        '-DMEFT_USE_LIBDEFLATE', '-l:liblz4.so.1', '-l:libzstd.so.1', '-l:libdeflate.so.0']
    builds = []
    for label, source in [('baseline', WORK / 'runtime/baseline'), ('candidate', WORK / 'runtime/native')]:
        library = WORK / 'lib' / (label + '.so')
        command = ['cc', '-std=c11', '-O3', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
            str(source / 'meft_rows.c'), '-lz', *flags,
            '-fopt-info-vec-optimized=' + str(WORK / 'results' / (label + '-vectorization.log')), '-o', str(library)]
        subprocess.run(command, check=True)
        builds.append({'label': label, 'command': command, 'source_sha256': digest(source / 'meft_rows.c'),
            'header_sha256': digest(source / 'meft_rows.h'), 'library_sha256': digest(library)})
    manifest = {'commit': COMMIT, 'compiler': subprocess.check_output(['cc', '--version'], text=True),
        'builds': builds, 'benchmark_sha256': digest(WORK / 'runtime/tools/benchmark_native_bags.py'),
        'binding_sha256': digest(WORK / 'runtime/tools/native_row_store.py'), 'tables': []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest['tables'] = list(pool.map(lambda x: build_table(*x), [(m, d) for m in MODELS for d in ('float16', 'bfloat16')]))
    (WORK / 'results/build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=str(WORK / 'runtime'), CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cores = sorted(os.sched_getaffinity(0))[80::2]
    if len(cores) < 5:
        cores = sorted(os.sched_getaffinity(0))[:5]
    if len(cores) < 5:
        raise ValueError('five distinct CPU cores required')
    jobs = []
    for entry in manifest['tables']:
        model = entry['model']; dtype = entry['stored_dtype']
        command = [sys.executable, '-m', 'tools.benchmark_native_bags', '--table', entry['table'],
            '--codes', entry['codes'], '--traces', str(PRIOR / 'traces' / (model.split('-')[0] + '.npz')),
            '--baseline-library', str(WORK / 'lib/baseline.so'), '--library', str(WORK / 'lib/candidate.so'),
            '--output', str(WORK / 'results' / (model + '-' + dtype + '.json')), '--commit', COMMIT,
            '--stored-dtype', dtype, '--core', str(cores[len(jobs)])]
        jobs.append((model + '-' + dtype, command))
    model = MODELS[0]; bundle = ROOT / 'application-frontier/models' / model / 'bundles'
    command = [sys.executable, '-m', 'tools.benchmark_native_bags', '--table', str(bundle / 'native-zstd-sq-fp16/codes.meft'),
        '--codes', str(bundle / 'faiss-sq-fp16/codes.npy'), '--traces', str(PRIOR / 'traces/ag_news.npz'),
        '--baseline-library', str(WORK / 'lib/baseline.so'), '--library', str(WORK / 'lib/candidate.so'),
        '--output', str(WORK / 'results' / (model + '-float16-zstd.json')), '--commit', COMMIT,
        '--stored-dtype', 'float16', '--core', str(cores[4]), '--batch-sizes', '32', '--repeat', '3', '--rounds', '2', '--warmup', '1']
    jobs.append((model + '-float16-zstd', command))
    def run(job):
        label, command = job
        with (WORK / 'results' / (label + '.log')).open('w') as out:
            result = subprocess.run(command, cwd=WORK / 'runtime', env=env, stdout=out, stderr=subprocess.STDOUT)
        print('benchmarked', label, result.returncode, flush=True)
        return {'label': label, 'returncode': result.returncode, 'command': command}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(run, jobs))
    (WORK / 'results/run-status.json').write_text(json.dumps(results, indent=2) + '\n')
    if any(r['returncode'] for r in results):
        raise RuntimeError('benchmark failed; inspect preserved logs')


if __name__ == '__main__':
    main()
