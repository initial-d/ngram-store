"""Codec-byte identity and query lifetime fixtures; no performance evidence."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    from tools.faiss_row_codec import fit_codec, FaissRows, pack_codec_native
    from tools.row_codec_utils import close_reader
except ImportError:
    np = None

LIBRARY = os.environ.get('MEFT_ROWS_LIBRARY')
AVAILABLE = np is not None and importlib.util.find_spec('faiss') is not None and LIBRARY is not None


@unittest.skipUnless(AVAILABLE, 'FAISS and compiled native row library required')
class NativeFaissCodeTests(unittest.TestCase):
    def test_quantized_and_fp32_codes_keep_original_ids_and_decode_bytes(self):
        import faiss
        faiss.omp_set_num_threads(1)
        rows = np.arange(512 * 8, dtype='<f4').reshape(512, 8) / 1000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for codec in ('fp32', 'sq-fp16', 'sq4', 'pq-2x4', 'opq-2x4', 'rq-2x4'):
                source = root / codec
                fit_codec(source, rows, codec, rows, iterations=2, opq_iterations=2)
                original_codes = np.load(source / 'codes.npy')
                original_hash = hashlib.sha256(original_codes.tobytes()).hexdigest()
                for reordered, block_codec in ((r, c) for r in (False, True) for c in ('zlib', 'row-xor')):
                    out = root / (codec + '-native-' + str(reordered) + '-' + block_codec)
                    permutation = list(reversed(range(len(rows)))) if reordered else None
                    meta = pack_codec_native(source, out, native_library=LIBRARY,
                        row_order=permutation, max_block_bytes=256, chunk_rows=73,
                        predictor='none' if block_codec == 'row-xor' else 'xor', block_codec=block_codec)
                    self.assertTrue(meta['codes_storage']['original_ids_byte_exact'])
                    self.assertEqual(meta['codes_storage']['original_code_bytes_sha256'], original_hash)
                    with FaissRows(source) as direct, FaissRows(out, native_library=LIBRARY, cache_bytes=512) as native:
                        self.assertIsNone(native.codes)
                        for ids in ([42, 0, 42, 511], list(range(512)), []):
                            self.assertEqual(native.rows(ids).tobytes(), direct.rows(ids).tobytes())
                        decoded = np.concatenate([x for _, x in native.decoded(73)])
                        self.assertEqual(decoded.shape, rows.shape)
                        # OPQ inverse transforms can round differently with a
                        # different BLAS batch shape; compare identical schedules.
                        reference = np.concatenate([x for _, x in direct.decoded(73)])
                        self.assertEqual(decoded.tobytes(), reference.tobytes())
                        for bad in ([-1], [512], [1.5], [[1]]):
                            with self.assertRaises(ValueError): native.rows(bad)
                        close_reader(native)
                        with self.assertRaises(ValueError): native.rows([0])
                    self.assertFalse((out / 'codes.raw').exists())
                    if codec != 'fp32':
                        self.assertEqual((source / 'codec.faiss').read_bytes(), (out / 'codec.faiss').read_bytes())

    def test_dedup_mapping_is_retained_without_expanding_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / 'direct';out = root / 'native'
            rows = np.array([[1., 2.], [3., 4.]], dtype='<f4')
            fit_codec(source, rows, 'fp32', rows)
            mapping = np.array([1, 0, 1], dtype='u1')
            np.save(source / 'original-to-code.npy', mapping)
            pack_codec_native(source, out, native_library=LIBRARY)
            self.assertEqual((source / 'original-to-code.npy').read_bytes(), (out / 'original-to-code.npy').read_bytes())
            with FaissRows(out, native_library=LIBRARY) as reader:
                self.assertEqual(reader.rows(mapping).tobytes(), rows[mapping].tobytes())
                self.assertEqual(reader.metadata['rows'], 2)

    def test_native_reader_rejects_changed_row_shape_and_codec_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / 'direct';out = root / 'native'
            rows = np.array([[1., 2.], [3., 4.]], dtype='<f4')
            fit_codec(source, rows, 'fp32', rows)
            pack_codec_native(source, out, native_library=LIBRARY)
            meta = json.loads((out / 'codec.json').read_text());meta['rows'] = 3
            (out / 'codec.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'native code shape'):
                FaissRows(out, native_library=LIBRARY)
            with self.assertRaises(ValueError):
                pack_codec_native(source, root / 'bad', native_library=LIBRARY, chunk_rows=0)


if __name__ == '__main__':
    unittest.main()
