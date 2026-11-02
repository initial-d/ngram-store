"""Incidence certificates and real optimizer differential tests, not synthetic evidence."""
import tempfile
import unittest
import hashlib
import json
import os
import subprocess
from pathlib import Path

try:
    import numpy as np
    from tools.cotraining_groups import BagPartition, CertifiedBagPlan, CertifiedBagRows, grouped_embedding_bag
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(np is not None, 'optional NumPy required')
class PartitionTest(unittest.TestCase):
    def test_fused_counts_match_scalar_certificates_and_native_plan_order(self):
        import itertools
        from collections import Counter
        # Group IDs need not follow representative original-ID order.
        p = BagPartition([1, 0, 2, 0, 1], 'noncontiguous fixture', 1)
        bags = [[i for i, n in enumerate(counts) for _ in range(n)]
                for counts in itertools.product(range(3), repeat=5)]
        compiled = p._compiled_groups(iter(bags))
        expected = [p.certifies(b) for b in bags]
        np.testing.assert_array_equal(compiled['accepted'], expected)
        safe_bags = [b for b, safe in zip(bags, expected) if safe]
        plan = CertifiedBagPlan(p, iter(safe_bags))
        reps = p.representatives()
        ids, weights, offsets = [], [], [0]
        for bag in safe_bags:
            for row, count in sorted(Counter(int(reps[i]) for i in bag).items()):
                ids.append(row)
                weights.append(count)
            offsets.append(len(ids))
        self.assertEqual(plan.ids.tobytes(), np.asarray(ids, dtype='<u8').tobytes())
        self.assertEqual(plan.weights.tobytes(), np.asarray(weights, dtype='<f4').tobytes())
        self.assertEqual(plan.offsets.tobytes(), np.asarray(offsets, dtype='<u8').tobytes())
        self.assertEqual(plan.denominators.tolist(), [max(1, len(b)) for b in safe_bags])
        self.assertEqual(CertifiedBagPlan(p, []).offsets.tolist(), [0])
        for bag in ([True], [-1], [5], [.5], [[0]]):
            with self.assertRaises(ValueError):
                CertifiedBagPlan(p, [bag])

    def test_batch_certificate_matches_exhaustive_scalar_queries(self):
        import itertools
        p = BagPartition.fit([[0, 1], [2, 3], [4]], 5)
        bags = [[i for i, n in enumerate(counts) for _ in range(n)]
                for counts in itertools.product(range(3), repeat=5)]
        expected = [p.certifies(bag) for bag in bags]
        np.testing.assert_array_equal(p.certifies_many(bags), expected)
        self.assertEqual(p.certifies_many([]).shape, (0,))
        np.testing.assert_array_equal(p.certifies_many([[], []]), [True, True])

    def test_batch_certificate_padding_invalid_and_noncontiguous_members(self):
        p = BagPartition.fit([[0, 5000], [1]], 5002)
        bags = [[0, 5000], [0], [5000, 0, 0], [], [1]]
        np.testing.assert_array_equal(p.certifies_many(bags), [p.certifies(b) for b in bags])
        protected = BagPartition.fit([[0, 1, 2]], 3, protected=[0])
        np.testing.assert_array_equal(protected.certifies_many([[0, 1, 2, 0], [0], [0, 1]], padding_idx=0), [True, True, False])
        np.testing.assert_array_equal(protected.certifies_many([[0, 1, 2]], padding_idx=np.int64(0)), [True])
        for bag in ([True], [-1], [5002], [.5], [[0]]):
            with self.assertRaises(ValueError):
                p.certifies_many([bag])
        with self.assertRaises(ValueError):
            p.certifies_many([[]], padding_idx=True)

    def test_multiplicity_other_hosts_and_empty_rows_do_not_merge(self):
        bags = [[0, 1, 2, 3, 3], [0, 4, 5]]
        p = BagPartition.fit(bags, 7, protected=[0])
        self.assertEqual(p.mapping[1], p.mapping[2])
        self.assertNotEqual(p.mapping[1], p.mapping[3])
        self.assertNotEqual(p.mapping[6], p.mapping[1])
        self.assertTrue(all(p.certifies(b) for b in bags))
        self.assertFalse(p.certifies([1]))
        self.assertFalse(p.certifies([1, 1, 2]))
        self.assertTrue(p.certifies([1, 1, 2, 2]))
        self.assertTrue(p.certifies([]))
        p2 = BagPartition.fit([[1, 2], [1]], 3)
        self.assertNotEqual(p2.mapping[1], p2.mapping[2])

    def test_projection_preserves_only_certified_bag_sums(self):
        p = BagPartition.fit([[0, 1, 2], [3, 4]], 5)
        rows = np.asarray([[1, 0], [0, 1], [2, 3], [7, 0], [1, 2]], dtype=np.float64)
        canonical = p.means(rows)[p.mapping]
        for bag in [[0, 1, 2], [3, 4], [0, 1, 2, 0, 1, 2]]:
            np.testing.assert_allclose(canonical[bag].sum(0), rows[bag].sum(0), rtol=1e-15, atol=1e-15)
        self.assertFalse(p.certifies([0, 3]))
        self.assertFalse(np.array_equal(canonical[[0, 3]].sum(0), rows[[0, 3]].sum(0)))
        self.assertFalse(np.array_equal(canonical, rows))
        reps = p.representatives()
        np.testing.assert_array_equal(reps[reps], reps)

    def test_save_roundtrip_and_invalid_inputs(self):
        p = BagPartition.fit([[0, 1]], 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partition.npz'
            p.save(path)
            loaded = BagPartition.load(path)
            self.assertEqual(loaded.summary(), p.summary())
            self.assertFalse(loaded.mapping.flags.writeable)
            self.assertEqual(loaded.mapping_sha256,
                             hashlib.sha256(loaded.mapping.astype('<u8').tobytes()).hexdigest())
            for array in (loaded.mapping, loaded.sizes):
                with self.assertRaises(ValueError):
                    array.flags.writeable = True
        for bags in ([[True]], [[-1]], [[3]], [[.5]]):
            with self.assertRaises(ValueError):
                BagPartition.fit(bags, 3)
        with self.assertRaises(ValueError):
            BagPartition([0, 2, 2], '', 1)
        with self.assertRaises(ValueError):
            p.means(np.full((3, 2), np.nan))

    def test_compiled_schedule_reduces_duplicates_and_rejects_partial_groups(self):
        p = BagPartition.fit([[0, 1, 2], [0, 3, 4]], 5, protected=[0])
        plan = CertifiedBagPlan(p, [[0, 1, 2], [3, 4, 3, 4], []], padding_idx=0)
        self.assertEqual(plan.summary()['original_occurrences'], 6)
        self.assertEqual(plan.summary()['group_occurrences'], 2)
        self.assertEqual(plan.offsets.tolist(), [0, 1, 2, 2])
        self.assertEqual(plan.weights.tolist(), [2, 4])
        with self.assertRaisesRegex(ValueError, 'uncertified'):
            CertifiedBagPlan(p, [[1]])

    def test_rejected_query_has_a_sparse_algebraic_counterexample(self):
        import itertools
        partition = BagPartition.fit([[0, 1], [2, 3], [4]], 5)
        for counts in itertools.product(range(3), repeat=5):
            bag = [i for i, count in enumerate(counts) for _ in range(count)]
            explanation = partition.explain_query(bag)
            expected = counts[0] == counts[1] and counts[2] == counts[3]
            self.assertEqual(explanation['certified'], expected)
            if expected:
                self.assertIsNone(explanation['counterexample'])
            else:
                witness = explanation['counterexample']
                source = np.zeros((5, 1), dtype=np.float64)
                source[witness['positive_row']] = 1
                source[witness['negative_row']] = -1
                canonical = partition.means(source)[partition.mapping]
                self.assertEqual(float(source[bag].sum()), witness['source_sum'])
                self.assertEqual(float(canonical[bag].sum()), witness['projected_sum'])
                self.assertGreater(witness['source_sum'], 0)
                self.assertEqual(partition.mapping[witness['positive_row']],
                                 partition.mapping[witness['negative_row']])

    def test_query_diagnostic_handles_noncontiguous_members_and_invalid_ids(self):
        partition = BagPartition.fit([[0, 5000], [1]], 5002)
        incomplete = partition.explain_query([0])
        self.assertEqual(incomplete['counterexample']['negative_row'], 5000)
        unequal = partition.explain_query([5000, 0, 0])
        self.assertEqual(unequal['violation']['reason'], 'unequal multiplicity')
        self.assertEqual(unequal['counterexample']['positive_row'], 0)
        for bag in ([True], [-1], [5002], [.5], [[0]]):
            with self.assertRaises(ValueError):
                partition.explain_query(bag)

    def test_native_weighted_schedule_empty_bags_and_artifact_integrity(self):
        from tools.native_row_store import pack_native
        p = BagPartition.fit([[0, 1, 2], [0, 3, 4]], 5, protected=[0])
        rows = np.asarray([[0, 0], [1, 0], [0, 1], [2, 2], [4, 0]], dtype='<f4')
        canonical = p.means(rows)[p.mapping]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            library = tmp / 'librows.so'
            root = Path(__file__).resolve().parents[1]
            subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-fPIC', '-shared',
                            str(root / 'native/meft_rows.c'), '-lz', '-o', str(library)], check=True)
            source = tmp / 'source.f32'
            canonical.tofile(source)
            pack_native(source, tmp / 'table.men', 8, representatives=p.representatives().tolist())
            p.save(tmp / 'domain.npz')
            manifest = {'partition': p.summary(),
                        'artifact_sha256': hashlib.sha256((tmp / 'table.men').read_bytes()).hexdigest()}
            (tmp / 'manifest.json').write_text(json.dumps(manifest))
            with CertifiedBagRows(tmp, library) as native:
                bags = [[0, 1, 2], [0, 3, 4, 3, 4], []]
                actual = native.aggregate(bags, padding_idx=0)
                np.testing.assert_allclose(actual, [rows[b].sum(0) for b in bags], atol=1e-7)
                np.testing.assert_allclose(native.aggregate(bags, mean=True, padding_idx=0),
                    actual / np.asarray([2, 4, 1])[:, None], atol=1e-7)
                with self.assertRaisesRegex(ValueError, 'uncertified'):
                    native.aggregate([[1]])
                other = BagPartition.fit([[0, 1], [2, 3]], 5, protected=[0])
                with self.assertRaisesRegex(ValueError, 'different partition'):
                    native.evaluate(CertifiedBagPlan(other, [[0, 1]]))
            manifest['artifact_sha256'] = 'bad'
            (tmp / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                CertifiedBagRows(tmp, library)

    def test_public_exporter_checks_rows_and_cleans_failed_output(self):
        from tools.cotraining_groups import export_certified_bags
        from unittest.mock import patch
        p = BagPartition.fit([[0, 1]], 2)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'export'
            with patch('tools.native_row_store.pack_native', side_effect=OSError('injected')):
                with self.assertRaisesRegex(OSError, 'injected'):
                    export_certified_bags(np.ones((2, 2), dtype='<f4'), p, output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, 'float32'):
                export_certified_bags(np.ones((2, 2), dtype=np.float64), p, output)
            output.mkdir()
            (output / 'keep').write_text('original')
            with self.assertRaises(FileExistsError):
                export_certified_bags(np.ones((2, 2), dtype='<f4'), p, output)
            self.assertEqual((output / 'keep').read_text(), 'original')

    def test_raw_block_option_keeps_original_bits_and_default_artifact(self):
        from tools.native_row_store import NativeRows, pack_native
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            library = tmp / 'librows.so'
            root = Path(__file__).resolve().parents[1]
            subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-fPIC', '-shared',
                            str(root / 'native/meft_rows.c'), '-lz', '-o', str(library)], check=True)
            # Compressible bytes plus signed zero/nonfinite payloads: row identity
            # is a byte contract, independent of how numbers would compare.
            raw = bytes.fromhex('000000800100c07f') * 64
            source = tmp / 'raw'
            source.write_bytes(raw)
            first = pack_native(source, tmp / 'default', 8, predictor='xor', deduplicate=False)
            pack_native(source, tmp / 'explicit', 8, predictor='xor', deduplicate=False, compress=True)
            self.assertEqual((tmp / 'default').read_bytes(), (tmp / 'explicit').read_bytes())
            result = pack_native(source, tmp / 'no-codec', 8, predictor='xor', deduplicate=False, compress=False)
            self.assertGreater(first['compressed_blocks'], 0)
            self.assertEqual(result['compressed_blocks'], 0)
            with NativeRows(tmp / 'no-codec', library) as native:
                self.assertEqual(native.rows(range(64)), raw)


@unittest.skipUnless(torch is not None and np is not None, 'optional NumPy/PyTorch required')
class TrainingTest(unittest.TestCase):
    def test_factory_releases_source_table_instead_of_retaining_a_class_closure(self):
        import gc
        import weakref
        source = torch.ones(5, 8)
        reference = weakref.ref(source)
        partition = BagPartition.fit([[0, 1, 2], [3, 4]], 5)
        model = grouped_embedding_bag(source, partition)
        del source
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(model.weight.shape, (2, 8))
        torch.testing.assert_close(model(torch.tensor([[0, 1, 2]])), torch.full((1, 8), 3.))

    def test_sparse_sgd_preserves_bag_function_and_group_mean_trajectory(self):
        # Several distinct initial rows, repeated incidence, and changing targets.
        torch.manual_seed(19)
        bags = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 0], [0, 1, 2, 3]])
        p = BagPartition.fit(bags.tolist(), 6, protected=[0])
        initial = torch.randn(6, 7, dtype=torch.float64) * .01
        initial[0].zero_()
        dense = torch.nn.EmbeddingBag.from_pretrained(initial.clone(), freeze=False,
                    mode='sum', sparse=True, padding_idx=0)
        tied = grouped_embedding_bag(initial, p, padding_idx=0)
        wrong = grouped_embedding_bag(initial, p, padding_idx=0, scale_gradient=False)
        optimizers = [torch.optim.SGD(m.parameters(), lr=.02) for m in (dense, tied, wrong)]
        initial_offset = initial[1] - initial[2]
        targets = torch.randn(12, 3, 7, dtype=torch.float64)
        for target in targets:
            for model, optimizer in zip((dense, tied, wrong), optimizers):
                optimizer.zero_grad(set_to_none=True)
                (model(bags) - target).square().sum().backward()
                optimizer.step()
            torch.testing.assert_close(dense(bags), tied(bags), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(tied.weight, torch.from_numpy(p.means(dense.weight.detach().numpy())),
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dense.weight[1] - dense.weight[2], initial_offset, rtol=1e-12, atol=1e-12)
        self.assertFalse(torch.allclose(dense(bags), wrong(bags)))
        with self.assertRaisesRegex(ValueError, 'uncertified'):
            tied(torch.tensor([[1, 0]]))

    def test_sparse_gradient_multiplicity_and_padding_guard(self):
        p = BagPartition.fit([[0, 1, 1, 2, 2]], 3, protected=[0])
        model = grouped_embedding_bag(torch.ones(3, 2), p, padding_idx=0)
        model(torch.tensor([[0, 1, 1, 2, 2]])).sum().backward()
        gradient = model.weight.grad.coalesce()
        torch.testing.assert_close(gradient.values(), torch.full((1, 2), 2.))
        with self.assertRaisesRegex(ValueError, 'singleton'):
            grouped_embedding_bag(torch.ones(3, 2), p, padding_idx=1)


if __name__ == '__main__':
    unittest.main()
