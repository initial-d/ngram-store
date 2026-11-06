import math
import unittest
from tools.analyze_tinyengram_joint import paired_nll


class JointAnalysisTest(unittest.TestCase):
    def windows(self,values):
        return [{'window_index':i,'targets':2,'source_nll':v} for i,v in enumerate(values)]

    def test_paired_gain_and_inverse_with_uneven_losses(self):
        full=self.windows([[1.,3.],[2.,4.]])
        baseline=self.windows([[1.2,3.2],[2.2,4.2]])
        gain=paired_nll(full,baseline,resamples=100)
        self.assertAlmostEqual(gain['baseline_relative_ppl_excess'],math.expm1(.2))
        self.assertAlmostEqual(gain['baseline_minus_full_nll_95ci'][0],.2)
        inverse=paired_nll(baseline,full,resamples=100)
        self.assertLess(inverse['baseline_minus_full_nll_95ci'][1],0)
        equal=paired_nll(full,full,resamples=100)
        self.assertEqual(equal['baseline_minus_full_nll_95ci'],[0.,0.])

    def test_reject_unpaired_windows_and_missing_targets(self):
        full=self.windows([[1.,2.],[3.,4.]])
        with self.assertRaises(ValueError):paired_nll(full,list(reversed(full)),resamples=10)
        short=self.windows([[1.],[3.,4.]])
        with self.assertRaises(ValueError):paired_nll(full,short,resamples=10)
