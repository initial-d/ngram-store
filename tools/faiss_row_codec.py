"""Optional FAISS standalone codecs for ORIGINAL-ID lookup, not ANN search.

The serialized index contains only the trained codec (ntotal=0). Per-row codes
are stored separately, so queries decode selected codes rather than reconstruct
the full table. This is an evaluation adapter, not a new production format.
"""
import json
from pathlib import Path
import time

import numpy as np


DEFAULT_CODECS = (
    'fp32', 'sq-fp16', 'sq8', 'cluster-8', 'pq-8x4', 'pq-8x8',
    'pq-16x8', 'opq-8x8', 'rq-2x8',
)


def make_index(name, dim, *, seed=42, iterations=20, opq_iterations=10):
    import faiss
    if name == 'sq-fp16':
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16)
    if name == 'sq8':
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit)
    if name == 'cluster-8':
        # One full-vector centroid ID: actual vector clustering, not scalar rounding.
        name = 'pq-1x8'
    family, shape = name.split('-', 1)
    m, bits = map(int, shape.split('x'))
    if m < 1 or not 1 <= bits <= 8 or (family != 'rq' and dim % m):
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
                'iterations': iterations, 'opq_iterations': opq_iterations,
                'training_seconds': training_seconds,
                'encoding_seconds': time.perf_counter() - start}
    (directory / 'codec.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


class FaissRows:
    def __init__(self, directory):
        import faiss
        directory = Path(directory)
        self.metadata = json.loads((directory / 'codec.json').read_text())
        self.codes = np.load(directory / 'codes.npy', mmap_mode='r', allow_pickle=False)
        self.index = None if self.metadata['codec'] == 'fp32' else faiss.read_index(str(directory / 'codec.faiss'))
        if self.codes.shape[0] != self.metadata['rows']:
            raise ValueError('row count mismatch')
        if self.index is not None and self.index.ntotal:
            raise ValueError('codec unexpectedly includes duplicated index vectors')

    def rows(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in 'iu'
                or ids.min() < 0 or ids.max() >= len(self.codes))):
            raise ValueError('one-dimensional valid original row IDs required')
        if not len(ids):
            return np.empty((0, self.metadata['dim']), dtype=np.float32)
        gathered = np.ascontiguousarray(self.codes[ids.astype(np.int64)])
        return gathered if self.index is None else self.index.sa_decode(gathered)

    def decoded(self, chunk_rows=4096):
        for begin in range(0, len(self.codes), chunk_rows):
            yield begin, self.rows(np.arange(begin, min(begin + chunk_rows, len(self.codes))))
