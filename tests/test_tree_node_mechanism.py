"""Independent common-update and initialization-residual fixtures."""
import unittest
try:
    import numpy as np
    from tools.cotraining_groups import BagPartition
    from tools.analyze_tree_node_mechanism import residual_diagnostics
except ImportError:
    np = None


@unittest.skipIf(np is None, 'optional NumPy required')
class MechanismTests(unittest.TestCase):
    def test_identical_updates_preserve_initial_difference_without_forcing_cosine(self):
        partition = BagPartition.fit([[0, 1], [0, 1], [2]], 3)
        initial = np.array([[1., 0.], [-1., 0.], [0., 2.]])
        update = np.array([[0., 10.], [0., 10.], [1., 1.]])
        result = residual_diagnostics(initial, initial + update, partition)
        self.assertEqual(result['groups'], 2)
        self.assertEqual(result['rows_in_nonsingleton_groups'], 2)
        self.assertEqual(result['initial_residual_norm'], 2. ** .5)
        self.assertEqual(result['final_residual_norm'], 2. ** .5)
        self.assertEqual(result['group_update_dispersion_relative'], 0.)
        self.assertEqual(result['residual_change_relative_to_initial'], 0.)
        # The same row relationship exists before training with cosine -1;
        # only the common update's size/direction makes final cosine high.
        final = initial + update
        cosine = final[0] @ final[1] / (np.linalg.norm(final[0]) * np.linalg.norm(final[1]))
        self.assertGreater(cosine, .98)

    def test_unequal_multiplicity_does_not_certify_common_updates(self):
        partition = BagPartition.fit([[0, 0, 1], [0, 1]], 2)
        result = residual_diagnostics(np.eye(2), np.eye(2) + [[0., 2.], [0., 1.]], partition)
        self.assertEqual(result['groups'], 2)
        self.assertEqual(result['rows_in_nonsingleton_groups'], 0)


if __name__ == '__main__':
    unittest.main()
