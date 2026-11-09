"""Frozen original-ID row codecs as bounded-staging multi-head embeddings."""
import threading
import time
import torch
import numpy as np


class CodecMultiHeadEmbedding(torch.nn.Module):
    """Stage only requested FP32 decoded rows; never materialize the full table.

    The reader owns its mmap/native/FAISS resources. Codes and codebooks stay
    immutable. CPU decoding and synchronous device copies are deliberate;
    this adapter adds no gradient, CUDA graph or asynchronous overlap contract.
    """
    def __init__(self, reader, head_sizes, device, *, staging_rows=4096, profile=False):
        super().__init__()
        if not head_sizes or any(type(n) is not int or n < 1 for n in head_sizes):
            raise ValueError('positive head sizes required')
        if type(staging_rows) is not int or staging_rows < 1:
            raise ValueError('positive staging row count required')
        self.reader = reader
        self.head_sizes = tuple(head_sizes)
        self.num_heads = len(head_sizes)
        self.embedding_dim = reader.metadata['dim']
        self.device = torch.device(device)
        self.staging_rows = staging_rows
        offsets = np.cumsum([0, *head_sizes[:-1]], dtype=np.int64)
        self.register_buffer('offsets', torch.from_numpy(offsets))
        self._cpu_offsets = torch.from_numpy(offsets.copy())
        self._cpu_limits = torch.tensor(head_sizes, dtype=torch.int64)
        self._closed = False
        self._lock = threading.Lock()
        self.profile = profile
        self.stats = {'calls': 0, 'requested_rows': 0, 'staged_bytes': 0,
            'maximum_chunk_rows': 0, 'decode_seconds': 0., 'transfer_seconds': 0.}
        physical = reader.metadata['rows']
        if reader.mapping is None:
            if physical != sum(head_sizes): raise ValueError('reader logical row domain differs from heads')
        elif len(reader.mapping) != sum(head_sizes):
            raise ValueError('code mapping does not cover original heads')

    @torch.no_grad()
    def forward(self, input_ids):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('codec memory is single-worker; concurrent gather/close unsupported')
        try:
            if self._closed: raise ValueError('codec memory is closed')
            if input_ids.dtype not in (torch.int32, torch.int64) or input_ids.ndim < 1 or input_ids.shape[-1] != self.num_heads:
                raise ValueError('integer per-head ID tensor required')
            local = input_ids.detach().to(device='cpu', dtype=torch.int64)
            if torch.any(local < 0) or torch.any(local >= self._cpu_limits):
                raise ValueError('local head ID out of bounds')
            ids = (local + self._cpu_offsets).contiguous().reshape(-1).numpy()
            output = torch.empty((*input_ids.shape, self.embedding_dim), dtype=torch.float32, device=self.device)
            flat = output.reshape(-1, self.embedding_dim)
            self.stats['calls'] += 1
            self.stats['requested_rows'] += len(ids)
            for begin in range(0, len(ids), self.staging_rows):
                selected = ids[begin:begin + self.staging_rows]
                if self.reader.mapping is not None:
                    selected = self.reader.mapping[selected]
                started = time.perf_counter() if self.profile else 0.
                values = self.reader.rows(selected)
                if values.shape != (len(selected), self.embedding_dim) or values.dtype != np.float32:
                    raise ValueError('codec returned a different FP32 row shape')
                if self.profile:
                    self.stats['decode_seconds'] += time.perf_counter() - started
                    if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                    started = time.perf_counter()
                flat[begin:begin + len(selected)].copy_(torch.from_numpy(np.ascontiguousarray(values)))
                if self.profile:
                    if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
                    self.stats['transfer_seconds'] += time.perf_counter() - started
                self.stats['staged_bytes'] += values.nbytes
                self.stats['maximum_chunk_rows'] = max(self.stats['maximum_chunk_rows'], len(selected))
            return output
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('codec memory is single-worker; concurrent close unsupported')
        try:
            if not self._closed:
                self.reader.close()
                self._closed = True
        finally:
            self._lock.release()
