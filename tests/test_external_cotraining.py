"""Exact external construction, adversarial collisions and eviction order."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec('numpy'):
    import numpy as np
    from tools.cotraining_groups import BagPartition
    from tools.external_cotraining import ExternalBagPartition, _Sorter, _records, iter_bag_files
else:
    np = None


@unittest.skipUnless(np is not None, 'NumPy required')
class ExternalConstructionTests(unittest.TestCase):
    def test_single_bag_exceeds_payload_buffer_and_spills_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 320,000 payload bytes exceed the 131,072-byte small-bag buffer.
            expected = BagPartition.fit([range(20000)], 20000)
            stages = []
            result = BagPartition.fit_external([iter(range(20000))], 20000, Path(tmp) / 'domain',
                memory_bytes=8 << 20, progress=lambda stage, path: stages.append(stage))
            self.assertIn('bag-spill', stages)
            self.assertEqual(result.summary(), expected.summary())
    def test_collision_resolution_protected_rows_empty_bags_and_exact_numbering(self):
        class Collision:
            def update(self, block):
                pass

            def digest(self):
                return b'constant'

        bags = [[4, 1, 4, 1, 7], [], [2, 5, 8], [1, 4], [0, 3, 6], [2, 5]]
        for protected in ([], [1, 1, 6]):
            with tempfile.TemporaryDirectory() as tmp, \
                    patch('tools.external_cotraining._signature_hash', Collision):
                expected = BagPartition.fit(bags, 10, protected=protected)
                result = BagPartition.fit_external((iter(b) for b in bags), 10, Path(tmp) / 'domain',
                                                  memory_bytes=8 << 20, protected=iter(protected))
                self.assertEqual(result.summary(), expected.summary())
                self.assertEqual(result.materialize().mapping.tobytes(), expected.mapping.tobytes())
                result.save_npz(Path(tmp) / 'domain.npz')
                self.assertEqual(BagPartition.load(Path(tmp) / 'domain.npz').summary(), expected.summary())
                self.assertEqual(sorted(p.name for p in result.path.iterdir()),
                                 ['external.json', 'mapping.u64', 'sizes.u64'])

    def test_external_sort_multiple_merge_passes_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            sorter = _Sorter(Path(tmp) / 'sort', 2, 8 << 20)
            sorter.limit = 16  # Force more than one bounded-fan-in merge pass.
            values = [(i % 7, i % 3) for i in range(300, -1, -1)]
            for value in values:
                sorter.add(*value)
            with sorter.finish().open('rb') as stream:
                self.assertEqual(list(_records(stream, 2)), sorted(values))
            self.assertEqual(len(list(sorter.path.iterdir())), 1)

    def test_file_projection_matches_ordered_float64_reference_with_evictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bags = [[i, i + 100, i + 200] for i in range(100)]
            expected = BagPartition.fit(bags, 300)
            external = BagPartition.fit_external(bags, 300, root / 'domain', memory_bytes=8 << 20)
            # Deterministic arithmetic fixture; not compression evidence.
            rows = (np.arange(300 * 4096, dtype=np.float32) % 127).reshape(300, 4096)
            rows[0, :4] = [1e20, -0., 1., -1e20]
            rows[100, :4] = [-1e20, 0., 1e-10, 1e20]
            rows.tofile(root / 'source.f32')
            report = external.project_file(root / 'source.f32', 4096, root / 'means.f32',
                                           memory_bytes=8 << 20)
            self.assertGreater(report['evictions'], 0)
            self.assertEqual((root / 'means.f32').read_bytes(), expected.means(rows).tobytes())
            self.assertFalse(any(p.name.startswith('.means-') for p in root.iterdir()))

    def test_failures_cleanup_owned_outputs_and_preserve_existing_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bag in ([True], [-1], [2], [.5]):
                with self.assertRaises(ValueError):
                    BagPartition.fit_external([bag], 2, root / 'domain', memory_bytes=8 << 20)
                self.assertFalse((root / 'domain').exists())
            with self.assertRaises(ValueError):
                BagPartition.fit_external([], 2, root / 'domain')
            result = BagPartition.fit_external([[]], 2, root / 'domain')
            np.array([[1.], [np.nan]], dtype=np.float32).tofile(root / 'source.f32')
            with self.assertRaises(ValueError):
                result.project_file(root / 'source.f32', 1, root / 'means.f32')
            self.assertFalse((root / 'means.f32').exists())
            (root / 'existing').write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                result.save_npz(root / 'existing')
            with self.assertRaises(FileExistsError):
                result.project_file(root / 'source.f32', 1, root / 'existing')
            self.assertEqual((root / 'existing').read_bytes(), b'preserve')
            payload = bytearray((result.path / 'mapping.u64').read_bytes())
            payload[-1] ^= 1
            (result.path / 'mapping.u64').write_bytes(payload)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                ExternalBagPartition(result.path)

    def test_public_namespaces_load_without_torch(self):
        import ngram_store
        import meft_memory
        self.assertIs(ngram_store.ExternalBagPartition, meft_memory.ExternalBagPartition)
        with tempfile.TemporaryDirectory() as tmp:
            result = ngram_store.fit_external([[0, 1]], 2, Path(tmp) / 'domain')
            self.assertEqual(result.summary()['group_rows'], 1)

    def test_csr_contract_and_cli_preserve_the_certified_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.array([2, 0, 1, 2, 0, 1], dtype='<u8').tofile(root / 'ids')
            np.array([0, 0, 3, 6], dtype='<u8').tofile(root / 'offsets')
            bags = [list(b) for b in iter_bag_files(root / 'ids', root / 'offsets')]
            self.assertEqual(bags, [[], [2, 0, 1], [2, 0, 1]])
            rows = np.array([[1., 0.], [0., 2.], [5., -3.]], dtype='<f4')
            rows.tofile(root / 'rows')
            result = subprocess.run([sys.executable, '-m', 'ngram_store.cli', 'compact-bags',
                '--input', str(root / 'rows'), '--dim', '2', '--bag-ids', str(root / 'ids'),
                '--bag-offsets', str(root / 'offsets'), '--output', str(root / 'out'),
                '--memory-mib', '8'], text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            self.assertFalse(report['source_row_bytes_preserved'])
            reference = BagPartition.fit(bags, 3)
            self.assertEqual(BagPartition.load(root / 'out/domain.npz').summary(), reference.summary())
            self.assertEqual((root / 'out/means.f32').read_bytes(), reference.means(rows).tobytes())
            np.asarray([1], dtype='<u8').tofile(root / 'protected')
            subprocess.run([sys.executable, '-m', 'ngram_store.cli', 'compact-bags',
                '--input', str(root / 'rows'), '--dim', '2', '--bag-ids', str(root / 'ids'),
                '--bag-offsets', str(root / 'offsets'), '--protected-ids', str(root / 'protected'),
                '--output', str(root / 'protected-out'), '--memory-mib', '8'],
                text=True, capture_output=True, check=True)
            protected = BagPartition.fit(bags, 3, protected=[1])
            self.assertEqual(BagPartition.load(root / 'protected-out/domain.npz').summary(), protected.summary())
            for offsets in ([1, 6], [0, 4, 3, 6], [0, 7], [0, 5]):
                np.asarray(offsets, dtype='<u8').tofile(root / 'offsets')
                with self.assertRaises(ValueError):
                    [list(b) for b in iter_bag_files(root / 'ids', root / 'offsets')]
