"""Complete serving bundles, original-ID forwarding and certificate enforcement."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if importlib.util.find_spec('numpy'):
    import numpy as np
    from tools.cotraining_groups import BagPartition, CertifiedBagRows, export_certified_bags
    from tools.certified_store import compact_native
    from tools.external_cotraining import fit_external
else:
    np = None


@unittest.skipUnless(np is not None, 'NumPy required')
class CertifiedStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.temp.name) / 'librows.so'
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-fPIC', '-shared',
            str(Path(__file__).resolve().parents[1] / 'native/meft_rows.c'), '-lz', '-o', str(cls.library)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def inputs(self, root):
        # Arithmetic fixture only; compression evidence uses actual trained tables.
        rows = np.asarray([[1., -2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.], [11., 12.]], dtype='<f4')
        bags = [[4, 1, 4, 1], [], [0, 3]]
        rows.tofile(root / 'rows')
        np.asarray([4, 1, 4, 1, 0, 3], dtype='<u8').tofile(root / 'ids')
        np.asarray([0, 4, 4, 6], dtype='<u8').tofile(root / 'offsets')
        return rows, bags

    def test_full_bundle_matches_legacy_canonical_rows_and_query_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, bags = self.inputs(root)
            expected = BagPartition.fit(bags, len(rows))
            report = compact_native(root / 'rows', 2, root / 'ids', root / 'offsets', root / 'new', memory_bytes=8 << 20)
            self.assertEqual(sorted(p.name for p in (root / 'new').iterdir()), ['domain.npz', 'manifest.json', 'table.men'])
            self.assertEqual(report['complete_serving_bytes'], sum(p.stat().st_size for p in (root / 'new').iterdir()))
            export_certified_bags(rows, expected, root / 'old', native_library=self.library, block_rows=8, compress=False)
            self.assertEqual((root / 'new/table.men').read_bytes(), (root / 'old/table.men').read_bytes())
            with CertifiedBagRows(root / 'new', self.library) as new, CertifiedBagRows(root / 'old', self.library) as old:
                self.assertEqual(new.table.rows(list(range(len(rows)))), expected.means(rows)[expected.mapping].tobytes())
                queries = bags + [[2], [5], [1, 4], []]
                for mean in (False, True):
                    self.assertEqual(new.aggregate(queries, mean=mean).tobytes(), old.aggregate(queries, mean=mean).tobytes())
                with self.assertRaisesRegex(ValueError, 'uncertified'):
                    new.aggregate([[1]])

    def test_cli_build_inspect_queries_and_rejected_raw_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, bags = self.inputs(root)
            command = [sys.executable, '-m', 'ngram_store.cli']
            built = subprocess.run(command + ['compact-bags', '--native', '--input', str(root / 'rows'),
                '--dim', '2', '--bag-ids', str(root / 'ids'), '--bag-offsets', str(root / 'offsets'),
                '--output', str(root / 'new'), '--memory-mib', '8'], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            (root / 'queries.json').write_text(json.dumps(bags))
            for mode in ('sum', 'mean'):
                result = subprocess.run(command + ['query-bags', '--store', str(root / 'new'),
                    '--bags', str(root / 'queries.json'), '--mode', mode, '--native-library', str(self.library)],
                    check=True, capture_output=True, text=True)
                report = json.loads(result.stdout)
                self.assertTrue(report['certified'])
                np.testing.assert_allclose(report['vectors'], [rows[bag].sum(axis=0) / max(len(bag), 1)
                    if mode == 'mean' else rows[bag].sum(axis=0) for bag in bags])
            inspected = subprocess.run(command + ['inspect', '--store', str(root / 'new'), '--native-library', str(self.library)],
                                       capture_output=True, text=True)
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            for subcommand in ('query', 'query-bags'):
                refused = subprocess.run(command + [subcommand, '--store', str(root / 'new'),
                    '--ids', '1', '--native-library', str(self.library)], capture_output=True, text=True)
                self.assertEqual(refused.returncode, 2)
                self.assertIn('query-bags' if subcommand == 'query' else 'uncertified', refused.stderr)

    def test_export_failure_cleanup_existing_output_and_mean_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, bags = self.inputs(root)
            original = rows.tobytes()
            np.asarray([0, 999], dtype='<u8').tofile(root / 'offsets')
            with self.assertRaises(ValueError):
                compact_native(root / 'rows', 2, root / 'ids', root / 'offsets', root / 'new', memory_bytes=8 << 20)
            self.assertFalse((root / 'new').exists())
            self.assertEqual((root / 'rows').read_bytes(), original)
            external = fit_external(bags, len(rows), root / 'partition')
            expected = BagPartition.fit(bags, len(rows))
            expected.means(rows).tofile(root / 'means')
            with self.assertRaisesRegex(ValueError, 'means changed'):
                external.export_native(root / 'means', 2, root / 'table', expected_means_sha256='changed')
            self.assertFalse((root / 'table').exists())
            (root / 'table').write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                external.export_native(root / 'means', 2, root / 'table')
            self.assertEqual((root / 'table').read_bytes(), b'keep')
            self.assertFalse(any(p.name.startswith('.native-') for p in root.iterdir()))
