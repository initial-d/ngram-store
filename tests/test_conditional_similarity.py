import itertools
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'optional NumPy unavailable')
class ConditionalSimilarityTest(unittest.TestCase):
    def test_uniform_pair_index_matches_exhaustive_combinations(self):
        from tools.conditional_similarity import unrank_pair, structural_pairs
        for n in range(2, 45):
            self.assertEqual([unrank_pair(n, i) for i in range(n * (n - 1) // 2)],
                             list(itertools.combinations(range(n), 2)))
        keys = [f'abc{i:03}' for i in range(100)]
        a, total = structural_pairs(keys, 'prefix', 17)
        b, _ = structural_pairs(keys, 'prefix', 17)
        self.assertEqual(total, 450)
        self.assertEqual(len(set(map(tuple, a))), 17)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(all(keys[x][:-1] == keys[y][:-1] for x, y in a))

    def test_parent_sibling_relations_and_matching_preserve_metadata(self):
        from tools.conditional_similarity import structural_pairs, matched_controls, related, frequency_bins
        keys = ['ab', 'abc', 'abd', 'xbc', 'uvw', 'qrs', 'def', 'xy', 'bc']
        counts = np.array([8, 5, 7, 5, 5, 5, 5, 8, 8])
        parent, _ = structural_pairs(keys, 'parent')
        self.assertIn((0, 1), list(map(tuple, parent)))
        self.assertIn((8, 1), list(map(tuple, parent)))
        control = matched_controls(keys, counts, parent)
        bins = frequency_bins(counts)
        for (a, b), c in zip(parent, control):
            if c >= 0:
                self.assertEqual(len(keys[b]), len(keys[c]))
                self.assertEqual(bins[b], bins[c])
                self.assertFalse(related(keys[a], keys[c]))
                self.assertNotEqual(b, c)

    def test_angle_norm_zero_and_replacement_error_are_distinct(self):
        from tools.conditional_similarity import pair_metrics, summarize
        rows = np.array([[1, 0], [2, 0], [0, 0], [-1, 0], [0, 1]], dtype=np.float32)
        result = pair_metrics(rows, np.array([[0, 1], [0, 2], [0, 3], [0, 4]]))
        self.assertEqual(result['cosine'][0], 1)
        self.assertEqual(result['norm_ratio'][0], .5)
        self.assertEqual(result['replacement_relative_l2'][0], .5)
        self.assertTrue(np.isnan(result['cosine'][1]))
        self.assertEqual(result['cosine'][2], -1)
        self.assertEqual(result['cosine'][3], 0)
        summary = summarize(result)
        self.assertEqual(summary['valid_pairs'], 3)
        self.assertEqual(summary['undefined_zero_norm_pairs'], 1)
        self.assertEqual(summary['ge_0_99_pairs'], 1)
        self.assertEqual(summary['joint_0_99_pairs'], 0)

    def test_conditioning_detects_simpson_reversal_and_ties(self):
        from tools.conditional_similarity import conditional_trend, spearman
        # Across lengths high frequency has high similarity, but frequency is
        # negatively related to similarity within each length. Both are kept.
        x = np.r_[np.tile([1, 2], 30), np.tile([100, 200], 30)]
        y = np.r_[np.tile([.2, .1], 30), np.tile([.9, .8], 30)]
        groups = np.r_[np.full(60, 3), np.full(60, 8)]
        self.assertGreater(spearman(x, y), 0)
        result = conditional_trend(x, y, groups)
        self.assertAlmostEqual(result['weighted_spearman'], -1)
        self.assertEqual(result['measured_pairs'], 120)
        self.assertIsNone(spearman(np.ones(3), np.arange(3)))

    def test_frequency_multiplicity_and_center_sampling_reproduce_source(self):
        from tools.analyze_conditional_similarity import metadata
        from tools.train_ngram_similarity import prepare
        text = 'aaaa cat aaaa dog cat aaaa dog cat'
        settings = dict(min_count=1, vocab_limit=2, steps=3, batch_size=4)
        words, keys, counts, exposure, signatures, events = metadata(text, 4, settings, 7)
        w, g, sig, pool, *_ = prepare(text, 4, 2, 1)
        self.assertEqual((words, keys), (w, g))
        i = keys.index('aa')
        # aaaa appears three times, with aa multiplicity three per word.
        self.assertEqual(counts[i], 9)
        rng = np.random.default_rng(7)
        reference = np.zeros(len(words), dtype=np.int64)
        for _ in range(3):
            np.add.at(reference, pool[rng.integers(len(pool), size=4), 0], 1)
        self.assertTrue(np.array_equal(events, reference))
        self.assertEqual(exposure[i], events[words.index('aaaa')] * 3)


if __name__ == '__main__':
    unittest.main()
