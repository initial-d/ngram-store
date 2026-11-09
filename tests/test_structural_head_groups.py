"""Do not invent prefix/suffix sharing across independent model heads."""
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from tools.tree_node_quality import structural_mapping
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy and Torch required')
class StructuralHeadTests(unittest.TestCase):
    def test_partial_keys_and_separate_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            common = Path(directory)
            rows = np.array([[0., 0.], [1., 1.], [1., 1.], [0., 0.], [1., 1.]], dtype=np.float32)
            groups = [{'start': 0, 'size': 3, 'keys': {'1': 'ab', '2': 'abc'}},
                      {'start': 3, 'size': 2, 'keys': {'1': 'abcd'}}]
            (common / 'gram-key-groups.json').write_text(json.dumps(groups))
            mapping, order = structural_mapping(rows, common,
                {'planner': 'reference', 'cosine': .99, 'norm_ratio': .99})
            np.testing.assert_array_equal(mapping, [0, 1, 1, 3, 4])
            self.assertEqual(sorted(order), list(range(5)))
            groups[1]['start'] = 2
            (common / 'gram-key-groups.json').write_text(json.dumps(groups))
            with self.assertRaises(ValueError):
                structural_mapping(rows, common, {'planner': 'reference', 'cosine': .99, 'norm_ratio': .99})


if __name__ == '__main__': unittest.main()
