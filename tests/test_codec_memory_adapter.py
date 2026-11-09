"""Original head offsets, mapped multiplicity and staging lifetime fixtures."""
import unittest
try:
    import numpy as np
    import torch
    from tools.codec_memory_adapter import CodecMultiHeadEmbedding
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch required')
class CodecMemoryTests(unittest.TestCase):
    def test_head_offsets_repeated_ids_mapping_and_empty_queries(self):
        class Reader:
            metadata = {'rows': 2, 'dim': 2}
            mapping = np.array([1, 0, 1])
            calls = []
            closed = False
            def rows(self, ids):
                self.calls.append(len(ids))
                return np.array([[1., 2.], [3., 4.]], dtype=np.float32)[ids]
            def close(self): self.closed = True
        reader = Reader()
        adapter = CodecMultiHeadEmbedding(reader, [2, 1], 'cpu', staging_rows=2)
        ids = torch.tensor([[[1, 0], [0, 0], [1, 0]]])
        expected = torch.tensor([[[[1., 2.], [3., 4.]], [[3., 4.], [3., 4.]], [[1., 2.], [3., 4.]]]])
        self.assertTrue(torch.equal(adapter(ids), expected))
        self.assertEqual(max(reader.calls), 2)
        self.assertEqual(adapter(torch.empty((1, 0, 2), dtype=torch.int64)).shape, (1, 0, 2, 2))
        for invalid in (torch.tensor([[-1, 0]]), torch.tensor([[0, 1]]), torch.tensor([[0., 0.]]), torch.tensor([[0]])):
            with self.assertRaises(ValueError): adapter(invalid)
        adapter.close();adapter.close()
        self.assertTrue(reader.closed)
        with self.assertRaises(ValueError): adapter(ids)


if __name__ == '__main__':
    unittest.main()
