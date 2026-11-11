import unittest
import numpy as np
from tools.large_ngram_application import document_keys, key_view, exact_ids, hash_ids, batch_csr


class CanonicalNgramTest(unittest.TestCase):
    def test_exact_order_boundary_duplicates_and_oov_do_not_alias(self):
        keys = document_keys([1, 2, 1], 4)
        self.assertEqual(keys.tolist(), [[1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0],
                                       [1, 2, 0, 0], [2, 1, 0, 0], [1, 2, 1, 0]])
        registry = np.unique(key_view(keys))
        ids = exact_ids(keys, registry)
        self.assertEqual(ids[0], ids[2])
        self.assertEqual(len(np.unique(ids)), 5)
        unknown = document_keys([1, (1 << 32) - 1], 4)
        self.assertEqual(exact_ids(unknown, registry)[1:].tolist(), [0, 0])
        self.assertTrue(np.array_equal(registry[ids - 1], key_view(keys)))

    def test_hash_is_stable_bounded_and_preserves_key_order(self):
        keys = document_keys([1, 2, 1], 4)
        for buckets in (1, 7, 1024):
            ids = hash_ids(keys, buckets)
            self.assertTrue(np.all((ids > 0) & (ids <= buckets)))
            self.assertEqual(ids[0], ids[2])
            self.assertTrue(np.array_equal(hash_ids(keys[::-1], buckets), ids[::-1]))

    def test_ragged_batches_keep_empty_documents_duplicates_and_mapping(self):
        ids = np.array([1, 2, 1, 3]); ptr = np.array([0, 3, 3, 4])
        flat, offsets = batch_csr(ids, ptr, [2, 1, 0], np.array([0, 5, 6, 7]))
        self.assertEqual(flat.tolist(), [7, 5, 6, 5])
        self.assertEqual(offsets.tolist(), [0, 1, 1, 4])
