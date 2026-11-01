"""Algebra/serialization fixtures only, not learned-vector performance evidence."""
import importlib.util
import copy
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    from tools.cotraining_groups import BagPartition
    from tools.benchmark_faiss_cotraining import schedule, evaluate, objective, fidelity
    from tools.faiss_row_codec import fit_codec, FaissRows
    from tools.analyze_faiss_cotraining import merge_reports
except ImportError:
    np = None


class ArrayReader:
    def __init__(self, rows):
        self.data = rows
        self.metadata = {'rows': len(rows), 'dim': rows.shape[1]}

    def rows(self, ids):
        return self.data[ids]


@unittest.skipIf(np is None, 'NumPy required')
class QueryContractTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([[1., 2.], [3., 0.], [5., -2.]], dtype=np.float32)
        self.words = np.array([[.5, -.5]], dtype=np.float32)
        self.partition = BagPartition([0, 0, 1], 'algebra-fixture', 1)
        self.bags = [[0, 1, 2, 2], [0], [], [0, 0, 1, 1]]
        self.word_ids = [0, -1, -1, -1]

    def test_certified_means_and_hybrid_fallback(self):
        p = self.partition
        reference = evaluate(ArrayReader(self.source), schedule(self.bags, self.word_ids), self.words, batch_bags=2)
        strict = schedule(self.bags, self.word_ids, p)
        self.assertEqual(strict['accepted'].tolist(), [True, False, True, True])
        canonical = ArrayReader(p.means(self.source))
        actual = evaluate(canonical, strict, self.words, batch_bags=2)
        np.testing.assert_array_equal(actual[strict['accepted']], reference[strict['accepted']])
        hybrid = schedule(self.bags, self.word_ids, p, fallback=True)
        np.testing.assert_array_equal(evaluate(canonical, hybrid, self.words, self.source[:2], batch_bags=1), reference)
        self.assertTrue(hybrid['accepted'].all())
        self.assertEqual(fidelity(reference, actual, strict['accepted'])['global_relative_l2'], 0.)

    def test_dedup_multiplicity_and_empty_boundaries(self):
        plan = schedule([[], [0, 0, 2], [], [1], []], [-1] * 5)
        self.assertEqual(plan['ids'].tolist(), [0, 2, 1])
        self.assertEqual(plan['weights'].tolist(), [2., 1., 1.])
        values = evaluate(ArrayReader(self.source), plan, self.words, batch_bags=5)
        np.testing.assert_array_equal(values[[0, 2, 4]], np.zeros((3, 2)))
        np.testing.assert_allclose(values[1], (2 * self.source[0] + self.source[2]) / 3)
        with self.assertRaises(ValueError):
            schedule([[0]], [])

    def test_objective_ties_and_stability(self):
        data = {'pairs': np.array([[0, 0]]), 'negative': np.array([[0, 0]])}
        scores = objective(np.array([[1000., 0.]]), np.array([[1000., 0.]]), data)
        self.assertTrue(np.isfinite(scores['loss']).all())
        self.assertEqual(scores['discrimination'][0], .5)

    def test_encoded_byte_mapping_has_no_incidence_restriction(self):
        original = np.array([[1., 2.], [1., 2.], [3., 4.]], dtype=np.float32)
        compact = ArrayReader(original[[0, 2]])
        bags, words = [[0], [1], [0, 1, 2]], [-1, -1, -1]
        mapped = schedule(bags, words, code_mapping=np.array([0, 0, 1]))
        self.assertTrue(mapped['accepted'].all())
        np.testing.assert_array_equal(evaluate(compact, mapped, self.words),
            evaluate(ArrayReader(original), schedule(bags, words), self.words))

    def test_combined_selection_uses_validation_and_rejects_changed_traces(self):
        def variant(name, size, validation, test):
            return {'name': name, 'model_bytes': size,
                    'known_word_aggregate_fidelity': {'global_relative_l2': .01},
                    'quality': {'validation': {'loss_relative_change': validation, 'discrimination_change': 0.},
                                'test': {'loss_relative_change': test, 'discrimination_change': 0.}}}
        run = {'order': 4, 'seed': 3, 'checkpoint_sha256': 'fixture', 'partition': {'mapping': 'fixture'},
               'training_original_ids_sha256': 'fixture', 'evaluation': {'sha256': 'fixture'},
               'traces': {'known': 'fixture'}, 'variants': [variant('direct-fp32', 1000, 0., 0.)],
               'failures': [], 'selection': {'0.001': {'direct': None, 'group': None, 'hybrid': None}}}
        base = {'status': 'completed', 'settings': {'max_discrimination_drop': .005, 'max_aggregate_error': .05}, 'runs': [run]}
        other = copy.deepcopy(base)
        other['runs'][0]['variants'] = [variant('direct-sq8', 100, .0001, .1),
                                        variant('direct-sq4', 50, .1, 0.)]
        combined = merge_reports(base, other)
        picked = combined['runs'][0]['selection']['0.001']['direct']
        self.assertEqual(picked['name'], 'direct-sq8')
        self.assertFalse(picked['test_quality_budget_passed'])
        other['runs'][0]['traces'] = {'known': 'changed'}
        with self.assertRaises(ValueError):
            merge_reports(base, other)


@unittest.skipIf(np is None or importlib.util.find_spec('faiss') is None, 'NumPy/FAISS required')
class CodecPersistenceTests(unittest.TestCase):
    def test_standalone_selected_id_decode_and_no_index_vectors(self):
        import faiss
        faiss.omp_set_num_threads(1)
        # Deterministic finite numerical fixture, not a compression-quality corpus.
        source = np.arange(512 * 8, dtype=np.float32).reshape(512, 8) / 1000
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('fp32', 'sq-fp16', 'sq8', 'sq6', 'sq4', 'pq-2x4', 'opq-2x4', 'rq-2x4'):
                with self.subTest(codec=name):
                    path = Path(tmp) / name
                    fit_codec(path, source, name, source, iterations=2, opq_iterations=2)
                    reader = FaissRows(path)
                    all_rows = np.concatenate([x for _, x in reader.decoded(chunk_rows=73)])
                    # OPQ inverse rotation uses BLAS; batch shape can change FP32 rounding.
                    np.testing.assert_allclose(reader.rows([42, 0, 42, 511]), all_rows[[42, 0, 42, 511]], rtol=1e-6, atol=1e-7)
                    self.assertEqual(reader.rows([]).shape, (0, 8))
                    if reader.index is not None:
                        self.assertEqual(reader.index.ntotal, 0)
                    else:
                        np.testing.assert_array_equal(all_rows, source)
                    with self.assertRaises(ValueError):
                        reader.rows([-1])
                    with self.assertRaises(ValueError):
                        reader.rows([512])


if __name__ == '__main__':
    unittest.main()
