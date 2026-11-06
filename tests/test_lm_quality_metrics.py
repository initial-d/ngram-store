"""Alignment, KL direction and paired weighting for actual-model evaluation."""
import math
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

from tools.lm_quality_metrics import memory_qualification


class QualificationTest(unittest.TestCase):
    def test_branch_effect_alone_does_not_qualify_identity(self):
        positive = {"relative_ppl_change": .02, "bootstrap": {"delta_nll_95ci": [.001, .03]}}
        negative = {"relative_ppl_change": -.02, "bootstrap": {"delta_nll_95ci": [-.03, -.001]}}
        self.assertFalse(memory_qualification({"memory-off": positive, "head-mean": negative}))
        self.assertTrue(memory_qualification({"memory-off": positive, "head-mean": positive}))
        weak = {"relative_ppl_change": .002, "bootstrap": {"delta_nll_95ci": [.001, .003]}}
        self.assertFalse(memory_qualification({"memory-off": positive, "head-mean": weak}))


@unittest.skipUnless(torch is not None, "optional PyTorch and NumPy required")
class NextTokenMetricsTest(unittest.TestCase):
    def test_shift_and_final_position_exclusion(self):
        from tools.lm_quality_metrics import next_token_metrics
        a = torch.tensor([[[0., 2.], [2., 0.], [0., 0.]]])
        b = a.clone()
        b[:, -1] = torch.tensor([100., -100.])
        ids = torch.tensor([[0, 1, 0]])
        m = next_token_metrics(a, b, ids)
        self.assertEqual(m["targets"], 2)
        self.assertEqual(m["source_correct"], 2)
        self.assertEqual(m["top1_changes"], 0)
        self.assertEqual(m["source_to_candidate_kl"], [0., 0.])
        self.assertAlmostEqual(m["source_nll"][0], math.log1p(math.exp(-2)))
        self.assertFalse(m["logits_byte_exact"])

    def test_kl_direction_matches_analytic_distribution(self):
        from tools.lm_quality_metrics import next_token_metrics
        a = torch.tensor([[[math.log(.9), math.log(.1)], [0., 0.]]], dtype=torch.float64)
        b = torch.tensor([[[math.log(.5), math.log(.5)], [0., 0.]]], dtype=torch.float64)
        m = next_token_metrics(a, b, torch.tensor([[1, 0]]))
        expected = .9 * math.log(.9/.5) + .1 * math.log(.1/.5)
        self.assertAlmostEqual(m["source_to_candidate_kl"][0], expected)
        self.assertEqual(m["candidate_nll"], [math.log(2)])

    def test_identical_logits_and_invalid_targets(self):
        from tools.lm_quality_metrics import next_token_metrics
        a = torch.zeros(1, 3, 2)
        ids = torch.tensor([[0, 1, 0]])
        self.assertTrue(next_token_metrics(a, a, ids)["logits_byte_exact"])
        for bad in (torch.tensor([[0, 2, 0]]), torch.tensor([[0., 1., 0.]])):
            with self.assertRaises(ValueError):
                next_token_metrics(a, a, bad)
        a[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            next_token_metrics(a, a, ids)

    def test_bootstrap_weights_targets_and_is_repeatable(self):
        from tools.lm_quality_metrics import paired_window_interval, summarize_windows
        def window(n):
            return {"targets": n, "source_nll": [1.] * n, "candidate_nll": [1.02] * n,
                    "source_to_candidate_kl": [.001] * n, "top1_changes": 0,
                    "source_correct": n, "candidate_correct": n,
                    "logits_byte_exact": False, "max_absolute_logit_delta": .1}
        windows = [window(2), window(7)]
        a = paired_window_interval(windows)
        self.assertEqual(a, paired_window_interval(windows))
        for bound in a["delta_nll_95ci"]:
            self.assertAlmostEqual(bound, .02)
        summary = summarize_windows(windows)
        self.assertEqual(summary["targets"], 9)
        self.assertAlmostEqual(summary["relative_ppl_change"], math.expm1(.02))


if __name__ == "__main__":
    unittest.main()
