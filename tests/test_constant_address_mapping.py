"""Independent controls must not depend on key identity or retain its index."""
import unittest
try:
    import numpy as np
    import torch
    from tools.counterfact_models import ConstantAddressMapping, HeadConstantEmbedding
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Qwen/Transformers and Torch dependencies required')
class ConstantAddressTests(unittest.TestCase):
    def test_token_identity_changes_neither_addresses_nor_constant_vectors(self):
        mapping = ConstantAddressMapping(12, 2)
        source = torch.nn.Module()
        source.num_heads = 2; source.embedding_dim = 2
        source.register_buffer('offsets', torch.tensor([0, 3]))
        source.embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[0., 0.], [3., 6.], [6., 3.], [4., 2.], [8., 6.]]))
        constants = HeadConstantEmbedding(source)
        a = mapping.hash(np.array([[1, 2, 3], [99, 0, 7]]))[12]
        b = mapping.hash(np.array([[91, 8, 9], [3, 2, 1]]))[12]
        self.assertEqual(a.shape, (2, 3, 2))
        self.assertEqual(a.tobytes(), b.tobytes())
        values = constants(torch.from_numpy(a))
        self.assertEqual(values.shape, (2, 3, 2, 2))
        torch.testing.assert_close(values[0, 0], torch.tensor([[3., 3.], [6., 4.]]))
        self.assertEqual(sum(p.numel() for p in constants.parameters()), 4)
        self.assertFalse(hasattr(mapping, 'vocabulary'))
        self.assertFalse(hasattr(mapping, 'source'))


if __name__ == '__main__':
    unittest.main()
