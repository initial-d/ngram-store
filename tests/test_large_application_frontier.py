import argparse
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from tools.large_application_frontier import (bag_logits, bernoulli_upper, publish_job,
        select_sources, stratified_harm_upper)
    from tools.benchmark_large_application import StagedClassifier, document_batches
    from tools.tree_node_quality import digest, load, write


class ArrayRows:
    def __init__(self, values): self.values = values; self.calls = []
    def rows(self, ids): self.calls.append(ids.copy()); return self.values[ids]


@unittest.skipUnless(np is not None, 'optional NumPy research dependency required')
class FrontierTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('MEFT_ROWS_LIBRARY'), 'compiled native row library required')
    def test_native_application_accepts_numpy_original_ids_and_retains_strict_validation(self):
        from tools.large_application_frontier import ApplicationRows
        from tools.native_row_store import pack_native
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source.raw'
            rows = np.array([[0., 0.], [1., -2.], [3., 4.]], dtype='<f4')
            source.write_bytes(rows.tobytes())
            pack_native(source, root / 'rows.meft', 8, 2, False)
            reader = ApplicationRows({'kind': 'native', 'directory': str(root),
                'library': os.environ['MEFT_ROWS_LIBRARY'], 'dim': 2})
            try:
                for ids in (np.array([2, 0, 1, 2], dtype=np.int64), np.array([], dtype=np.int64), [2, 1]):
                    np.testing.assert_array_equal(reader.rows(ids), rows[ids])
                for ids in (np.array([-1]), np.array([3]), np.array([1.5]), np.array([True])):
                    with self.assertRaises(IndexError): reader.rows(ids)
            finally: reader.close()

    def test_empty_oov_duplicate_and_mapped_bags_match_independent_means(self):
        rows = np.array([[99., 99.], [1., 2.], [-3., 4.]], dtype=np.float32)
        ids = np.array([1, 0, 1, 2, 0]); ptr = np.array([0, 3, 3, 4, 5])
        classifier = {'weight': np.array([[2., 1.], [1., -2.]], dtype=np.float32),
                      'bias': np.array([.5, -.5], dtype=np.float32)}
        actual = bag_logits(ArrayRows(rows), classifier, ids, ptr, [0, 1, 2, 3])
        mean = np.array([[1., 2.], [0., 0.], [-3., 4.], [0., 0.]])
        np.testing.assert_array_equal(actual, mean @ classifier['weight'].T + classifier['bias'])
        mapped = bag_logits(ArrayRows(rows), classifier, ids, ptr, [0, 1, 2, 3], np.array([0, 2, 1]))
        np.testing.assert_array_equal(mapped, mean[[2, 1, 0, 3]] @ classifier['weight'].T + classifier['bias'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch dependency required')
    def test_staged_chunk_bound_and_gpu_schedule_match_cpu_semantics(self):
        rows = np.array([[100., 100.], [1., -2.], [3., 4.]], dtype=np.float32)
        reader = ArrayRows(rows)
        classifier = {'weight': np.array([[2., 3.]], dtype=np.float32), 'bias': np.array([.25], dtype=np.float32)}
        engine = StagedClassifier(reader, classifier, 'cpu', chunk_rows=2)
        documents = [np.array([1, 0, 2, 1, 2]), np.array([], dtype=np.int64), np.array([0])]
        actual = engine(documents).numpy()
        hidden = np.array([(rows[1].astype(np.float64)*2 + rows[2]*2)/4, [0., 0.], [0., 0.]])
        expected = (hidden @ classifier['weight'].T + classifier['bias']).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(engine.maximum_chunk_rows, 2)
        self.assertTrue(all(len(x) <= 2 and np.all(x != 0) for x in reader.calls))
        self.assertEqual(engine.staged_bytes, 4 * 2 * 4)

    def test_document_batch_schedule_keeps_empty_last_partial_and_hash_mapping(self):
        ids = np.array([1, 2, 0, 3]); ptr = np.array([0, 2, 2, 3, 4]); labels = np.arange(4)
        batches = list(document_batches(ids, ptr, labels, 3, np.array([0, 5, 6, 7])))
        self.assertEqual([i for i, _ in batches], [0, 3])
        self.assertEqual([[x.tolist() for x in batch] for _, batch in batches], [[[5, 6], [], [0]], [[7]]])

    def test_kl_bound_matches_zero_case_and_dominates_observed_harm(self):
        self.assertAlmostEqual(bernoulli_upper(0, 1000, 20), 1 - math.exp(-math.log(400) / 1000))
        self.assertEqual(bernoulli_upper(1000, 1000, 20), 1.)
        for count in (1, 25, 500, 999):
            p = bernoulli_upper(count, 1000, 20); q = count / 1000
            self.assertGreater(p, q)
            kl = q * math.log(q/p) + (1-q) * math.log((1-q)/(1-p))
            self.assertAlmostEqual(kl, math.log(400) / 1000, places=11)
        with self.assertRaises(ValueError): bernoulli_upper(2, 1, 1)

    def test_stratification_and_family_correction_are_conservative(self):
        labels = np.repeat([0, 1], [900, 100]); base = np.ones(1000, dtype=bool); candidate = base.copy()
        candidate[-10:] = False
        upper, strata = stratified_harm_upper(labels, base, candidate, 10)
        self.assertGreater(upper, .01)
        self.assertEqual([r['documents'] for r in strata], [900, 100])
        larger, _ = stratified_harm_upper(labels, base, candidate, 100)
        self.assertGreater(larger, upper)

    def test_complete_job_bytes_include_serialized_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.json'
            publish_job(path, {'complete_model_bytes': 500, 'name': 'baseline'})
            result = load(path)
            self.assertEqual(result['required_job_metadata_bytes'], path.stat().st_size)
            self.assertEqual(result['complete_model_bytes'], 500 + path.stat().st_size)

    def test_registered_grid_selection_uses_calibration_and_retains_invalid_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); training = root / 'training'
            for dataset in ('ag_news', 'dbpedia_14'):
                for init in ('zero', 'uniform'):
                    for address, buckets in (('exact', 1048576), ('hash-matched', 1048576),
                                             ('hash-budget', 1048576), ('hash-budget', 4194304)):
                        for rate in (1., 3., 10.):
                            for seed in (3, 7, 11):
                                directory = training / f'{dataset}-{init}-{address}-{buckets}-{rate}-{seed}'
                                directory.mkdir(parents=True)
                                np.save(directory / 'rows.npy', np.zeros((3, 2), dtype=np.float32))
                                np.savez(directory / 'classifier.npz', weight=np.ones((2, 2)))
                                np.savez(directory / 'calibration.npz', labels=np.array([0, 1]))
                                invalid = dataset == 'dbpedia_14' and init == 'uniform' and address == 'exact' and rate == 10
                                report = {'status': 'incomplete-row-coverage' if invalid else 'complete',
                                    'epochs': [{}]*10, 'rows': 3, 'changed_rows': 1 if invalid else 2,
                                    'rows_sha256': digest(directory / 'rows.npy'), 'seed': seed, 'initialization': init,
                                    'addressing': address, 'settings': {'data': str(root / dataset), 'buckets': buckets, 'lr': rate},
                                    'calibration': {'mean_nll': 0 if rate == 10 else rate,
                                        'sha256': digest(directory / 'calibration.npz')}}
                                write(directory / 'training.json', report)
            out = root / 'selected.json'
            select_sources(argparse.Namespace(training=training, output=out))
            selected = load(out)
            self.assertEqual(len(selected['selected']), 48)
            self.assertEqual(len(selected['unqualified_training_cells']), 3)
            self.assertFalse(selected['test_used'])
            family = [e for e in selected['selected'] if e['family'][:3] == ['dbpedia_14', 'uniform', 'exact']]
            self.assertEqual([e['learning_rate'] for e in family], [1., 1., 1.])
            chosen = next(training.glob('ag_news-zero-exact-*-1.0-3'))
            (chosen / 'rows.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'fixture changed'):
                select_sources(argparse.Namespace(training=training, output=out))


if __name__ == '__main__': unittest.main()
