"""Optional FAISS standalone codecs for ORIGINAL-ID lookup, not ANN search.

The serialized index contains only the trained codec (ntotal=0). Per-row codes
are stored separately, so queries decode selected codes rather than reconstruct
the full table. This is an evaluation adapter, not a new production format.
"""
import json
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np


DEFAULT_CODECS = (
    'fp32', 'sq-fp16', 'sq8', 'sq6', 'sq4', 'cluster-8', 'cluster-9', 'cluster-10', 'cluster-11', 'pq-8x4', 'pq-8x8',
    'pq-16x8', 'pq-32x8', 'pq-64x8', 'opq-8x8', 'opq-32x8', 'opq-64x8',
    'rq-2x8', 'rq-4x8',
)


def make_index(name, dim, *, seed=42, iterations=20, opq_iterations=10):
    import faiss
    if name == 'sq-fp16':
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16)
    if name == 'sq8':
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
    if name == 'sq6':
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_6bit)
    if name == 'sq4':
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_4bit)
    if name.startswith('cluster-'):
        # One full-vector centroid ID: actual vector clustering, not scalar rounding.
        name = 'pq-1x' + name.split('-', 1)[1]
    family, shape = name.split('-', 1)
    m, bits = map(int, shape.split('x'))
    if m < 1 or not 1 <= bits <= 16 or (family != 'rq' and dim % m):
        raise ValueError('invalid codec shape or nondivisible dimension')
    if family in ('pq', 'opq'):
        pq = faiss.IndexPQ(dim, m, bits)
        pq.pq.cp.seed, pq.pq.cp.niter = seed, iterations
        if family == 'pq':
            return pq
        opq = faiss.OPQMatrix(dim, m)
        opq.niter = opq_iterations
        opq.niter_pq = opq.niter_pq_0 = iterations
        # OPQ holds a borrowed C++ pointer. Retain its Python owner until training
        # finishes; assigning a temporary can leave a dangling pointer.
        opq._training_pq_owner = faiss.ProductQuantizer(dim, m, bits)
        opq.pq = opq._training_pq_owner
        opq._training_pq_owner.cp.seed = seed
        return faiss.IndexPreTransform(opq, pq)
    if family == 'rq':
        rq = faiss.IndexResidualQuantizer(dim, m, bits)
        rq.rq.cp.seed, rq.rq.cp.niter = seed, iterations
        return rq
    raise ValueError('unknown codec: ' + name)


def fit_codec(directory, rows, name, train_rows, *, seed=42, iterations=20,
              opq_iterations=10, chunk_rows=4096):
    """Train on caller-supplied ACTUAL rows; encode all stored rows in chunks."""
    import faiss
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows, train_rows = np.asarray(rows), np.asarray(train_rows, dtype=np.float32)
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('nonempty finite row matrix required')
    dim = rows.shape[1]
    if train_rows.ndim != 2 or train_rows.shape[1] != dim or not len(train_rows):
        raise ValueError('matching nonempty training rows required')
    start = time.perf_counter()
    index = None if name == 'fp32' else make_index(name, dim, seed=seed,
                        iterations=iterations, opq_iterations=opq_iterations)
    if index is not None:
        index.train(np.ascontiguousarray(train_rows))
        if not index.is_trained or index.ntotal != 0:
            raise ValueError('standalone codec must be trained without stored index vectors')
        faiss.write_index(index, str(directory / 'codec.faiss'))
    training_seconds = time.perf_counter() - start
    code_size = dim * 4 if index is None else index.sa_code_size()
    dtype = '<f4' if index is None else 'uint8'
    width = dim if index is None else code_size
    codes = np.lib.format.open_memmap(directory / 'codes.npy', mode='w+', dtype=dtype,
                                     shape=(len(rows), width))
    start = time.perf_counter()
    for begin in range(0, len(rows), chunk_rows):
        x = np.ascontiguousarray(rows[begin:begin + chunk_rows], dtype=np.float32)
        codes[begin:begin + len(x)] = x if index is None else index.sa_encode(x)
    codes.flush()
    del codes
    metadata = {'codec': name, 'dim': dim, 'rows': len(rows), 'code_size': code_size,
                'training_rows': len(train_rows), 'seed': seed,
                'training_threads': faiss.omp_get_max_threads(),
                'iterations': iterations, 'opq_iterations': opq_iterations,
                'training_seconds': training_seconds,
                'encoding_seconds': time.perf_counter() - start}
    (directory / 'codec.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


class FaissRows:
    def __init__(self, directory, *, native_library=None, cache_bytes=0):
        import faiss
        directory = Path(directory)
        self.metadata = json.loads((directory / 'codec.json').read_text())
        self.native = None
        self.codes = None
        self.closed = False
        storage = self.metadata.get('codes_storage')
        if storage is None:
            self.codes = np.load(directory / 'codes.npy', mmap_mode='r', allow_pickle=False)
        else:
            from .native_row_store import NativeRows
            if storage['backend'] != 'native' or storage['file'] != 'codes.meft':
                raise ValueError('unsupported native code storage')
            self.native = NativeRows(directory / storage['file'], native_library, cache_bytes=cache_bytes)
            if self.native.header != {'n_rows': self.metadata['rows'], 'row_bytes': self.metadata['code_size']}:
                self.native.close()
                raise ValueError('native code shape mismatch')
        try:
            self.index = None if self.metadata['codec'] == 'fp32' else faiss.read_index(str(directory / 'codec.faiss'))
            if self.codes is not None and self.codes.shape[0] != self.metadata['rows']:
                raise ValueError('row count mismatch')
            if self.index is not None and self.index.ntotal:
                raise ValueError('codec unexpectedly includes duplicated index vectors')
            if self.index is not None and (self.index.d != self.metadata['dim']
                    or self.index.sa_code_size() != self.metadata['code_size']):
                raise ValueError('codec dimensions differ from stored codes')
        except BaseException:
            self.close()
            raise

    def rows(self, ids):
        if self.closed:
            raise ValueError('row codec is closed')
        ids = np.asarray(ids)
        if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in 'iu'
                or ids.min() < 0 or ids.max() >= self.metadata['rows'])):
            raise ValueError('one-dimensional valid original row IDs required')
        if not len(ids):
            return np.empty((0, self.metadata['dim']), dtype=np.float32)
        if self.native is None:
            gathered = np.ascontiguousarray(self.codes[ids.astype(np.int64)])
        else:
            ids = np.ascontiguousarray(ids, dtype='<u8')
            dtype = '<f4' if self.index is None else 'uint8'
            width = self.metadata['dim'] if self.index is None else self.metadata['code_size']
            gathered = np.empty((len(ids), width), dtype=dtype)
            self.native.gather_into(ids.ctypes.data, len(ids), gathered.ctypes.data, gathered.nbytes)
        return gathered if self.index is None else self.index.sa_decode(gathered)

    def decoded(self, chunk_rows=4096):
        for begin in range(0, self.metadata['rows'], chunk_rows):
            yield begin, self.rows(np.arange(begin, min(begin + chunk_rows, self.metadata['rows'])))

    def close(self):
        if not self.closed:
            if self.native is not None:
                self.native.close()
            if self.codes is not None:
                self.codes._mmap.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def pack_codec_native(source, output, *, native_library, max_block_bytes=8192,
                      row_order=None, chunk_rows=4096, block_rows=256, predictor='xor'):
    """Losslessly pack existing FAISS code bytes without refitting its codec.

    Original IDs and standalone codebooks stay fixed. This preserves the already
    quantized table, rather than removing quantization error. Row permutations
    retain the native forwarding map and its complete cost. The source is read
    in chunks; native descriptors and optional permutation maps still grow.
    """
    from .native_row_store import NativeRows, pack_native
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('positive chunk row count required')
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('fresh native codec output required')
    metadata = json.loads((source / 'codec.json').read_text())
    if metadata.get('codes_storage') is not None:
        raise ValueError('source must contain original standalone codes')
    codes = np.load(source / 'codes.npy', mmap_mode='r', allow_pickle=False)
    dtype = np.dtype('<f4' if metadata['codec'] == 'fp32' else 'uint8')
    width = metadata['dim'] if metadata['codec'] == 'fp32' else metadata['code_size']
    if codes.shape != (metadata['rows'], width) or codes.dtype != dtype:
        codes._mmap.close()
        raise ValueError('source code shape or dtype mismatch')
    output.mkdir(parents=True)
    staging = output / 'codes.raw'
    digest = hashlib.sha256()
    try:
        with staging.open('wb') as stream:
            for begin in range(0, len(codes), chunk_rows):
                data = np.ascontiguousarray(codes[begin:begin + chunk_rows]).tobytes()
                digest.update(data); stream.write(data)
        storage = pack_native(staging, output / 'codes.meft', metadata['code_size'],
            block_rows=block_rows, deduplicate=False, predictor=predictor,
            max_block_bytes=max_block_bytes, row_order=row_order)
        with NativeRows(output / 'codes.meft', native_library) as reader:
            for begin in range(0, len(codes), chunk_rows):
                end = min(begin + chunk_rows, len(codes))
                if reader.rows(list(range(begin, end))) != np.ascontiguousarray(codes[begin:end]).tobytes():
                    raise AssertionError('native packing changed original-ID code bytes')
        if metadata['codec'] != 'fp32':
            shutil.copyfile(source / 'codec.faiss', output / 'codec.faiss')
        if (source / 'original-to-code.npy').is_file():
            shutil.copyfile(source / 'original-to-code.npy', output / 'original-to-code.npy')
        published = {**metadata, 'codes_storage': {'backend': 'native', 'file': 'codes.meft',
            'original_code_bytes_sha256': digest.hexdigest(), 'original_ids_byte_exact': True,
            'max_block_bytes': max_block_bytes, 'storage': storage}}
        (output / 'codec.json').write_text(json.dumps(published, indent=2) + '\n')
        staging.unlink()
        return published
    finally:
        codes._mmap.close()
