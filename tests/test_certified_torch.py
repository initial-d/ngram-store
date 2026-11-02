"""Actual CPU PyTorch EmbeddingBag integration on a frozen certified table."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from torch.nn import functional as F
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'PyTorch/NumPy required')
class CertifiedTorchTests(unittest.TestCase):
    def test_cpu_fixed_and_ragged_bags_padding_and_rejection(self):
        from tools.certified_store import compact_native
        from tools.certified_torch import CertifiedEmbeddingBag
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / 'librows.so'
            subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-fPIC', '-shared',
                str(Path(__file__).resolve().parents[1] / 'native/meft_rows.c'), '-lz', '-o', str(library)], check=True)
            rows = np.asarray([[0, 0], [1, 2], [3, 4], [5, 6]], dtype='<f4')
            rows.tofile(root / 'rows')
            np.asarray([0, 1, 2, 3], dtype='<u8').tofile(root / 'ids')
            np.asarray([0, 3, 4], dtype='<u8').tofile(root / 'offsets')
            np.asarray([0], dtype='<u8').tofile(root / 'protected')
            compact_native(root / 'rows', 2, root / 'ids', root / 'offsets', root / 'store', protected_ids=root / 'protected')
            fixed = torch.tensor([[0, 1, 2], [3, 0, 0], [0, 0, 0]])
            flat = torch.tensor([0, 1, 2, 3])
            edges = torch.tensor([0, 3, 4, 4])
            for mode in ('sum', 'mean'):
                with CertifiedEmbeddingBag(root / 'store', mode=mode, padding_idx=0,
                                           include_last_offset=True, native_library=library) as adapter:
                    expected = F.embedding_bag(fixed, torch.from_numpy(rows), mode=mode, padding_idx=0)
                    self.assertTrue(torch.equal(adapter(fixed), expected))
                    self.assertTrue(torch.equal(adapter(flat, edges), expected))
                    self.assertFalse(adapter(fixed).requires_grad)
                    self.assertEqual(list(adapter.parameters()), [])
                    with self.assertRaisesRegex(ValueError, 'uncertified'):
                        adapter(torch.tensor([[1]]))
                    with self.assertRaisesRegex(ValueError, 'per-sample'):
                        adapter(fixed, per_sample_weights=torch.ones_like(fixed, dtype=torch.float32))
                    for bad in (torch.tensor([0.]), torch.tensor([True]), torch.ones((1, 1, 1), dtype=torch.int64)):
                        with self.assertRaises(ValueError):
                            adapter(bad)
                    with self.assertRaisesRegex(ValueError, 'final offset'):
                        adapter(flat, torch.tensor([0, 3]))
