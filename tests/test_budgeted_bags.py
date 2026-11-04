"""Source-specific error bounds, exact arithmetic oracles and budget refusal."""
from fractions import Fraction
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if importlib.util.find_spec('numpy'):
    import numpy as np
    from tools.budgeted_bags import BudgetedBagRows, build_error_bounds
    from tools.certified_store import compact_native
    from tools.cotraining_groups import BagPartition, CertifiedBagRows, export_certified_bags
else:
    np = None


@unittest.skipUnless(np is not None, 'NumPy required')
class BudgetedBagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.temp.name) / 'librows.so'
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-fPIC', '-shared',
            str(Path(__file__).resolve().parents[1] / 'native/meft_rows.c'), '-lz', '-o', str(cls.library)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, root, rows, bags):
        rows = np.asarray(rows, dtype='<f4')
        rows.tofile(root / 'source.f32')
        partition = BagPartition.fit(bags, len(rows))
        export_certified_bags(rows, partition, root / 'store', native_library=self.library, compress=False)
        report = build_error_bounds(root / 'source.f32', root / 'store', root / 'bounds',
                                    native_library=self.library, chunk_rows=1)
        return rows, partition, report

    def test_centered_bound_improves_partial_and_unequal_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, partition, report = self.build(root, [[1.], [3.], [5.], [7.]], [[0, 1, 2, 3]])
            self.assertEqual(report['sidecar_bytes'], sum(p.stat().st_size for p in (root / 'bounds').iterdir()))
            self.assertEqual(sorted(p.name for p in (root / 'bounds').iterdir()),
                             ['bounds.json', 'group-defect-l2.f64', 'group-total-l2.f64', 'row-l2.f64'])
            with BudgetedBagRows(root / 'store', root / 'bounds', self.library) as native:
                result = native.assess([[0, 1, 2], [0, 0, 1, 2, 3], [0, 1, 2, 3], []])
                self.assertLess(result.projection_l2_bounds[0], 3.000001)
                self.assertLess(result.projection_l2_bounds[1], 3.000001)
                self.assertLess(result.projection_l2_bounds[2], 1e-12)
                self.assertEqual(result.exact_domain.tolist(), [False, False, True, True])
                self.assertEqual(result.absolute_l2_bounds[-1], 0)
                self.assertEqual(result.relative_l2_bounds[-1], 0)
                # Triangle-only bound is 3+1+1=5 for the partial query.
                self.assertLess(result.projection_l2_bounds[0], 5.)
                accepted = native.aggregate([[0, 1, 2]], max_absolute_l2=3.001)
                self.assertEqual(accepted.vectors.tolist(), [[12.]])
                with self.assertRaisesRegex(ValueError, 'exceeds L2 error budget'):
                    native.aggregate([[0, 1, 2]], max_absolute_l2=2.999)
            with CertifiedBagRows(root / 'store', self.library) as strict:
                with self.assertRaisesRegex(ValueError, 'uncertified'):
                    strict.aggregate([[0, 1, 2]])

    def test_bounds_enclose_exact_fraction_source_sums_and_means(self):
        # Fixtures stress cancellation, mean rounding, subnormal FP32 values,
        # incomplete groups and unequal counts. These are not trained evidence.
        fixtures = [
            [[1.1, -1.2], [1.3, 1.4], [1.5, -1.6]],
            [[1e20, 1e-38], [-1e20, 3e-38], [1., -1e-38]],
            [[np.nextafter(np.float32(0), np.float32(1)), 1.], [0., 1.], [0., -2.]],
        ]
        for fixture in fixtures:
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows, _, _ = self.build(root, fixture, [[0, 1, 2]])
                bags = [[i for i, count in enumerate(counts) for _ in range(count)]
                        for counts in itertools.product(range(3), repeat=3)]
                with BudgetedBagRows(root / 'store', root / 'bounds', self.library) as native:
                    for mean in (False, True):
                        result = native.assess(bags, mean=mean)
                        for bid, bag in enumerate(bags):
                            source = [sum((Fraction(float(rows[row, j])) for row in bag), Fraction())
                                      / (max(len(bag), 1) if mean else 1) for j in range(rows.shape[1])]
                            error2 = sum((Fraction(float(result.vectors[bid, j])) - source[j]) ** 2
                                         for j in range(rows.shape[1]))
                            bound2 = Fraction(float(result.absolute_l2_bounds[bid])) ** 2
                            self.assertLessEqual(error2, bound2)
                            norm2 = sum(v*v for v in source)
                            relative = float(result.relative_l2_bounds[bid])
                            if np.isfinite(relative) and norm2:
                                self.assertLessEqual(error2, Fraction(relative)**2 * norm2)
                            if not norm2 and error2:
                                self.assertFalse(np.isfinite(relative))

    def test_cancellation_padding_empty_batches_and_explicit_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, [[1.], [-1.]], [[0, 1]])
            with BudgetedBagRows(root / 'store', root / 'bounds', self.library) as native:
                result = native.assess([[0]])
                self.assertTrue(np.isinf(result.relative_l2_bounds[0]))
                self.assertIsNone(result.summary()['relative_l2_bounds'][0])
                with self.assertRaisesRegex(ValueError, 'exceeds'):
                    native.aggregate([[0]], max_relative_l2=1.)
                native.aggregate([[0]], max_absolute_l2=1.001)
                with self.assertRaisesRegex(ValueError, 'exceeds'):
                    native.aggregate([[0]], max_absolute_l2=1.001, max_relative_l2=.5)
                padded = native.assess([[0, 1], [0], []], padding_idx=0)
                np.testing.assert_array_equal(padded.vectors, native.assess([[1], [], []]).vectors)
                self.assertEqual(padded.exact_domain.tolist(), [False, True, True])
                self.assertEqual(native.aggregate([], max_absolute_l2=0).vectors.shape, (0, 1))
                native.aggregate([[]], max_relative_l2=0)
                for budget in (-1., float('nan'), float('inf'), True):
                    with self.assertRaises(ValueError):
                        result.accepts(max_relative_l2=budget)
                with self.assertRaisesRegex(ValueError, 'explicit'):
                    native.aggregate([[0]])
                for bag in ([True], [0.5], [-1], [2]):
                    with self.assertRaises(ValueError):
                        native.assess([bag])

    def test_chunk_invariance_hash_binding_cleanup_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, _, _ = self.build(root, [[1.1], [1.3], [1.5]], [[0, 1, 2]])
            build_error_bounds(root / 'source.f32', root / 'store', root / 'second',
                               native_library=self.library, chunk_rows=128)
            for name in ('row-l2.f64', 'group-total-l2.f64', 'group-defect-l2.f64'):
                self.assertEqual((root / 'bounds' / name).read_bytes(), (root / 'second' / name).read_bytes())
            with self.assertRaises(FileExistsError):
                build_error_bounds(root / 'source.f32', root / 'store', root / 'bounds', native_library=self.library)
            (root / 'bounds/row-l2.f64').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'hash'):
                BudgetedBagRows(root / 'store', root / 'bounds', self.library)
            np.asarray([[float('nan')], [1.], [2.]], dtype='<f4').tofile(root / 'bad')
            with self.assertRaisesRegex(ValueError, 'finite'):
                build_error_bounds(root / 'bad', root / 'store', root / 'failed', native_library=self.library)
            self.assertFalse((root / 'failed').exists())
            np.asarray([0, 1, 2], dtype='<u8').tofile(root / 'ids')
            np.asarray([0, 3], dtype='<u8').tofile(root / 'offsets')
            compact_native(root / 'source.f32', 1, root / 'ids', root / 'offsets', root / 'new')
            rows[0, 0] = 9
            rows.tofile(root / 'changed')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                build_error_bounds(root / 'changed', root / 'new', root / 'mismatch', native_library=self.library)
            self.assertFalse((root / 'mismatch').exists())

    def test_cli_opt_in_gate_and_default_strict_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, [[1.], [1.001]], [[0, 1]])
            command = [sys.executable, '-m', 'ngram_store', 'query-bags', '--store', str(root / 'store'),
                       '--ids', '0', '--native-library', str(self.library)]
            strict = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(strict.returncode, 2)
            self.assertIn('uncertified', strict.stderr)
            args = ['--bounds', str(root / 'bounds'), '--max-relative-l2', '.001']
            approximate = subprocess.run(command + args, capture_output=True, text=True)
            self.assertEqual(approximate.returncode, 0, approximate.stderr)
            report = json.loads(approximate.stdout)
            self.assertFalse(report['certified'])
            self.assertTrue(report['approximation_opt_in'])
            self.assertFalse(report['bounds']['exact_domain'][0])
            self.assertLess(report['bounds']['relative_l2_bounds'][0], .001)
            for invalid in (args[:-1] + ['.00001'], ['--max-relative-l2', '1'], ['--bounds', str(root / 'bounds')]):
                result = subprocess.run(command + invalid, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
            built = subprocess.run([sys.executable, '-m', 'ngram_store', 'build-error-bounds',
                '--input', str(root / 'source.f32'), '--store', str(root / 'store'), '--output', str(root / 'cli-bounds'),
                '--native-library', str(self.library)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertEqual(json.loads(built.stdout)['format'], 'source-residual-l2-v1')

    def test_nonfinite_native_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build(root, [[np.finfo(np.float32).max]], [[0]])
            with BudgetedBagRows(root / 'store', root / 'bounds', self.library) as native:
                with self.assertRaisesRegex(ValueError, 'nonfinite native output'):
                    native.assess([[0, 0]])


if __name__ == '__main__':
    unittest.main()
