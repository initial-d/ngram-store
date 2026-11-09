"""Post-encoding fidelity fixture: a source guard is not an encoding guard."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import numpy as np
    from tools.audit_tree_node_rows import audit
except ImportError:
    np = None


@unittest.skipIf(np is None, 'optional NumPy required')
class RowAuditTests(unittest.TestCase):
    def test_final_encoding_can_violate_original_norm_gate_and_zero_cosine_is_undefined(self):
        source = np.array([[1., 0.], [1., 1.], [2., 0.]], dtype=np.float32)
        decoded = np.array([[.9, 0.], [1., 1.], [0., 0.]], dtype=np.float32)
        class Reader:
            mapping = None
            def __init__(self, job):
                pass
            def rows(self, ids):
                return decoded[ids]
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'source-rows.npy', source)
            np.save(root / 'source-occurrence-counts.npy', np.array([1, 2, 3]))
            (root / 'build.json').write_text(json.dumps({'decoded_row_sha256': hashlib.sha256(decoded.tobytes()).hexdigest()}))
            (root / 'job.json').write_text(json.dumps({'name': 'fixture', 'common': str(root), 'report': str(root / 'build.json'),
                'cosine': .99, 'norm_ratio': .99}))
            with patch('tools.audit_tree_node_rows.Reader', Reader), patch('tools.audit_tree_node_rows.environment', return_value={}):
                audit(root / 'job.json', root / 'result.json')
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['rows'], 3)
            self.assertEqual(result['undefined_zero_norm_rows'], 1)
            self.assertEqual(result['post_encoding_rows_outside_original_joint_gate'], 2)
            self.assertAlmostEqual(result['cosine']['minimum'], 1.)
            self.assertAlmostEqual(result['norm_ratio']['minimum'], .9, places=6)
            self.assertEqual(result['relative_l2']['maximum'], 1.)
            self.assertFalse(result['source_row_bytes_preserved'])


if __name__ == '__main__':
    unittest.main()
