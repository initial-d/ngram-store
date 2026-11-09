"""Independent scoring/domain/selection fixtures, not performance evidence."""
import unittest
try:
    import numpy as np
    from tools.tree_node_quality import (context_evaluation, strata_masks, paired_blocks,
        block_summary, bootstrap, quality_gate, select, string_ngrams, lookup_strings, plan_for)
except ImportError:
    np = None


@unittest.skipIf(np is None, 'optional NumPy required')
class TreeNodeQualityTests(unittest.TestCase):
    def test_all_oov_centers_preserve_original_context_positions(self):
        arrays, metadata = context_evaluation(['a', 'unknown', 'b'], ['a', 'b'],
            {'a': 0, 'unknown': 2, 'b': 1}, {'a': 3, 'b': 2}, 42)
        self.assertEqual(arrays['pairs'].tolist(), [[0, 1], [2, 0], [2, 1], [1, 0]])
        self.assertEqual(arrays['positions'].tolist(), [0, 1, 1, 2])
        self.assertEqual(metadata['pairs'], 4)
        self.assertEqual(metadata['registered_target_token_fraction'], 2 / 3)
        masks = strata_masks(arrays, np.array([0, 1, -1]), np.array([True, True, True]))
        self.assertEqual(masks['known'].tolist(), [True, False, False, True])
        self.assertEqual(int(masks['oov_nonempty'].sum()), 2)

    def test_negatives_include_collisions_and_empty_oov_stays_a_stratum(self):
        arrays, metadata = context_evaluation(['a', 'x', 'a'], ['a'], {'a': 0, 'x': 1}, {'a': 4}, 42)
        self.assertTrue((arrays['negative'] == 0).all())
        self.assertEqual(metadata['negative_target_collision_fraction'], 1.)
        masks = strata_masks(arrays, np.array([0, -1]), np.array([True, False]))
        self.assertEqual(int(masks['oov_empty'].sum()), 2)
        self.assertEqual(int(masks['oov_nonempty'].sum()), 0)

    def test_key_multiplicity_and_known_cache_do_not_change_oov_input(self):
        class Index:
            def lookup(self, keys, index='datrie'):
                return [{'aa': 0, '<a': 1, 'a>': 2}.get(k) for k in keys]
        strings = ['aaa', 'aa']
        bags, ids = lookup_strings(strings, Index(), {'aaa': 0}, 2)
        self.assertEqual(bags, [[1, 0, 0, 2], [1, 0, 2]])
        cached_bags, cached_ids = lookup_strings(strings, Index(), {'aaa': 0}, 2, True)
        self.assertEqual(cached_bags, [[], [1, 0, 2]])
        self.assertEqual(ids, cached_ids)
        class Reader:
            mapping = None
            metadata = {'rows': 3, 'dim': 2}
        ordinary = plan_for(bags, ids, Reader())
        cached = plan_for(bags, ids, Reader(), True)
        self.assertEqual(ordinary['denominators'].tolist(), [5., 3.])
        self.assertEqual(cached['denominators'].tolist(), [1., 3.])
        self.assertEqual(cached['ids'].tolist(), [0, 1, 2])
        self.assertEqual(string_ngrams('aaa', 2).count('aa'), 2)

    def test_paired_blocks_keep_counts_and_weighted_objective(self):
        baseline = {'loss': np.array([1., 3., 2.]), 'discrimination': np.array([1., .5, 0.])}
        candidate = {'loss': np.array([2., 3., 1.]), 'discrimination': np.array([.5, .5, .5])}
        blocks = paired_blocks(candidate, baseline, np.array([0, 1, 128]), np.ones(3, dtype=bool))
        self.assertEqual(blocks['count'], [2, 1])
        self.assertEqual(blocks['candidate_loss'], [5., 1.])
        summary = block_summary(blocks)
        self.assertEqual(summary['pairs'], 3)
        self.assertEqual(summary['loss_relative_change'], 0.)
        self.assertEqual(summary['discrimination_change'], 0.)
        empty = paired_blocks(candidate, baseline, np.array([0, 1, 128]), np.zeros(3, dtype=bool))
        self.assertIsNone(block_summary(empty)['loss_relative_change'])

    def test_single_block_bootstrap_is_exactly_the_paired_change(self):
        b = {'block_ids': [0], 'count': [2], 'candidate_loss': [4.2], 'baseline_loss': [4.],
             'candidate_discrimination': [1.], 'baseline_discrimination': [1.5]}
        result = bootstrap(b, repeats=50)
        np.testing.assert_allclose(result['loss_relative_change_95_percent_interval'], [.05, .05])
        self.assertEqual(result['discrimination_change_95_percent_interval'], [-.25, -.25])

    def test_selection_requires_oov_quality_and_uses_validation_only(self):
        def quality(all_loss, oov_loss):
            return {'all': {'pairs': 100, 'loss_relative_change': all_loss, 'discrimination_change': 0.},
                    'oov_nonempty': {'pairs': 10, 'loss_relative_change': oov_loss, 'discrimination_change': 0.}}
        candidates = [{'name': 'small', 'model_bytes': 10, 'validation': {'0': quality(0., .02)}},
                      {'name': 'valid', 'model_bytes': 20, 'validation': {'0': quality(.001, .001)}, 'test': 'deliberately unavailable'}]
        self.assertFalse(quality_gate(candidates[0]['validation']['0'], .001))
        self.assertEqual(select(candidates, .001, False)['name'], 'valid')
        candidates[1]['validation']['0']['oov_nonempty']['pairs'] = 0
        self.assertIsNone(select(candidates, .001, False))


if __name__ == '__main__':
    unittest.main()
