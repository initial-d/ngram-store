"""Reject scientifically misleading aggregation and inconsistent sufficient stats."""
import unittest

from tools.verify_memory_application import gate, metrics, paired


class EvidenceArithmeticTests(unittest.TestCase):
    def test_official_scores_weight_cases_not_prompt_count(self):
        cases = []
        for index, successes in enumerate(([True], [False] * 9)):
            groups = {}
            for group in ('rewrite', 'paraphrase', 'neighborhood'):
                groups[group] = [{'success': win, 'correct': win,
                    'target_new': 2. if win == (group == 'neighborhood') else 1.,
                    'target_true': 1. if win == (group == 'neighborhood') else 2.} for win in successes]
            cases.append({'case_id': index, 'groups': groups})
        result = metrics(cases)
        self.assertEqual(result['rewrite']['prompts'], 10)
        self.assertEqual(result['rewrite']['success'], .5)
        self.assertEqual(result['harmonic_score'], .5)
        cases[0]['groups']['neighborhood'][0]['success'] = False
        with self.assertRaises(AssertionError):
            metrics(cases)

    def test_an_oov_failure_cannot_hide_inside_pooled_quality(self):
        quality = {s: {'pairs': 10, 'loss_relative_change': 0., 'discrimination_change': 0.}
                   for s in ('all', 'oov_nonempty')}
        self.assertTrue(gate(quality, .001))
        quality['oov_nonempty']['loss_relative_change'] = .002
        self.assertFalse(gate(quality, .001))
        quality['oov_nonempty']['loss_relative_change'] = 0.
        quality['oov_nonempty']['discrimination_change'] = -.006
        self.assertFalse(gate(quality, .001))

    def test_subgroup_stats_must_partition_all_context_pairs(self):
        empty = {'block_ids': [], 'count': [], 'candidate_loss': [], 'baseline_loss': [],
                 'candidate_discrimination': [], 'baseline_discrimination': []}
        block = {'block_ids': [0], 'count': [2], 'candidate_loss': [2.], 'baseline_loss': [2.],
                 'candidate_discrimination': [1.], 'baseline_discrimination': [1.]}
        full = {'pairs': 2, 'sgns_loss_per_pair': 1., 'baseline_sgns_loss_per_pair': 1.,
                'context_discrimination': .5, 'baseline_context_discrimination': .5,
                'loss_relative_change': 0., 'discrimination_change': 0.}
        absent = {'pairs': 0, 'loss_relative_change': None, 'discrimination_change': None}
        blocks = {'all': block, 'known': block, 'oov_nonempty': empty, 'oov_empty': empty}
        quality = {'all': full, 'known': full, 'oov_nonempty': absent, 'oov_empty': absent}
        paired(blocks, quality, {'pairs': 2, 'tokens': 10})
        blocks['oov_nonempty'] = block; quality['oov_nonempty'] = full
        with self.assertRaises(AssertionError):
            paired(blocks, quality, {'pairs': 2, 'tokens': 10})


if __name__ == '__main__':
    unittest.main()
