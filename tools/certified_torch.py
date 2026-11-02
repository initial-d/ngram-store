"""Frozen CPU nn.Module for certified sum/mean EmbeddingBag integration."""
import numpy as np
import torch
from torch import nn

from .cotraining_groups import CertifiedBagRows


class CertifiedEmbeddingBag(nn.Module):
    """Use original IDs on an immutable native table; reject uncertified bags.

    CPU int32/int64 input only; sum/mean modes, no gradients/per-sample weights.
    Native FP64 reduction is not bit identity to PyTorch's FP32 reduction.
    """
    def __init__(self, path, *, mode='mean', padding_idx=None, include_last_offset=False,
                 native_library=None):
        super().__init__()
        if mode not in ('sum', 'mean'):
            raise ValueError('certified EmbeddingBag supports sum/mean')
        if type(include_last_offset) is not bool:
            raise ValueError('include_last_offset must be boolean')
        self.store = CertifiedBagRows(path, native_library)
        self.mode, self.padding_idx = mode, padding_idx
        self.include_last_offset = include_last_offset
        self.num_embeddings, self.embedding_dim = len(self.store.partition.mapping), self.store.dim
        self.eval()

    def forward(self, input, offsets=None, per_sample_weights=None):
        if per_sample_weights is not None:
            raise ValueError('per-sample weights need a separate certificate')
        if not isinstance(input, torch.Tensor) or input.device.type != 'cpu' or \
                input.dtype not in (torch.int32, torch.int64) or input.ndim not in (1, 2):
            raise ValueError('CPU int32/int64 input with one or two dimensions required')
        if input.ndim == 2:
            if offsets is not None:
                raise ValueError('offsets are only valid for one-dimensional input')
            bags = input.tolist()
        else:
            if not isinstance(offsets, torch.Tensor) or offsets.device.type != 'cpu' or \
                    offsets.dtype not in (torch.int32, torch.int64) or offsets.ndim != 1:
                raise ValueError('CPU integer offsets required for one-dimensional input')
            edges = offsets.tolist()
            if not edges or edges[0] != 0 or any(a > b for a, b in zip(edges, edges[1:])) or \
                    edges[-1] > input.numel():
                raise ValueError('offsets must start at zero, be nondecreasing and fit input')
            if self.include_last_offset:
                if edges[-1] != input.numel():
                    raise ValueError('final offset must equal input length')
            else:
                edges.append(input.numel())
            values = input.tolist()
            bags = [values[a:b] for a, b in zip(edges, edges[1:])]
        plan = self.store.compile(bags, padding_idx=self.padding_idx)
        return torch.from_numpy(self.store.evaluate(plan, mean=self.mode == 'mean'))

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def extra_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, mode={self.mode}, frozen=True'
