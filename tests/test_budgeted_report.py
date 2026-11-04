"""Reject incomplete or internally inconsistent public evidence ledgers."""
import copy
import itertools
import unittest

from tools.analyze_budgeted_bags import BUDGETS, summarize, validate


def fixture():
    # Corruption fixtures only, not trained benchmark evidence.
    bounds, errors, strict = [.0002, .005], [.0001, .004], [True, False]
    points = []
    for budget in BUDGETS:
        mask = [v <= budget for v in bounds]
        points.append({'budget': budget, 'accepted_nonempty': sum(mask), 'coverage': sum(mask)/2,
            'new_nonempty_vs_strict': sum(a and not s for a, s in zip(mask, strict)),
            'maximum_actual_relative_l2': max((e for a, e in zip(mask, errors) if a), default=None),
            'oracle_eligible_nonempty': sum(e <= budget for e in errors)})
    trace = {'queries': 2, 'nonempty': 2, 'empty': 0, 'strict_nonempty': 1, 'strict_coverage': .5,
        'all_absolute_and_relative_bounds_pass': True, 'strict_native_output_bytes_identical': True,
        'absolute_errors': errors, 'relative_errors': errors, 'absolute_l2_bounds': bounds,
        'relative_l2_bounds': bounds, 'strict_mask': strict, 'nonempty_mask': [True, True], 'points': points,
        'cost': {'full_assess_all_queries': {'p50_seconds': .1}, 'source_fp64_all_queries': {'p50_seconds': .001}}}
    runs = []
    for convention, order, seed in itertools.product(('legacy', 'average'), (4, 8, 12), (3, 7, 11)):
        runs.append({'convention': convention, 'order': order, 'seed': seed,
            'canonical_rows_identical': True, 'partition': {'original_rows': 2, 'group_rows': 1},
            'capacity': {'raw_ngram_fp32_bytes': 64, 'unchanged_word_output_registry_bytes': 16,
                'native_certified_bytes': 20, 'sidecar_bytes': 40, 'raw_complete_model_bytes': 80,
                'budgeted_complete_model_bytes': 76, 'complete_model_saving_vs_raw_fp32': 1-76/80,
                'source_fallback_retained': False},
            'traces': {'known': copy.deepcopy(trace), 'oov': copy.deepcopy(trace)},
            'trace_sha256': {'known': 'a'*64, 'oov': 'b'*64}})
    return {'status': 'passed', 'failures': [], 'budgets': BUDGETS, 'runs': runs, 'source_commit': 'c'*40}


class BudgetedReportTests(unittest.TestCase):
    def test_summary_and_complete_costs(self):
        summary = summarize(fixture())
        row = summary['conventions']['legacy'][0]
        self.assertEqual(row['strict_oov_coverage'], .5)
        self.assertEqual(row['oov_budget_coverage'][7], 1.)
        self.assertEqual(row['sidecar_bytes'], 40)

    def test_incomplete_duplicated_or_failed_runs_are_rejected(self):
        for change in ('duplicate', 'missing', 'failed'):
            report = fixture()
            if change == 'duplicate':
                report['runs'][-1] = copy.deepcopy(report['runs'][0])
            elif change == 'missing':
                report['runs'].pop()
            else:
                report['failures'] = [{'worker': 'failed'}]
            with self.assertRaises(ValueError):
                validate(report)

    def test_false_bounds_coverage_and_costs_are_rejected(self):
        for change in ('bound', 'coverage', 'omitted', 'fallback', 'strict'):
            report = fixture()
            run = report['runs'][0]
            if change == 'bound':
                run['traces']['oov']['absolute_errors'][0] = .1
            elif change == 'coverage':
                run['traces']['oov']['points'][0]['coverage'] = 1.
            elif change == 'omitted':
                run['capacity']['budgeted_complete_model_bytes'] -= 40
            elif change == 'fallback':
                run['capacity']['source_fallback_retained'] = True
            else:
                run['traces']['oov']['strict_native_output_bytes_identical'] = False
            with self.assertRaises(ValueError):
                validate(report)


if __name__ == '__main__':
    unittest.main()
