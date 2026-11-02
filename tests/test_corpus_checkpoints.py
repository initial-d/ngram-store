"""Corpus-prefix integrity fixtures; not learned-vector quality evidence."""
import tempfile
import unittest
from pathlib import Path
import zipfile

try:
    from tools.train_corpus_checkpoints import corpus_prefix
except ImportError:
    corpus_prefix = None


@unittest.skipIf(corpus_prefix is None, 'NumPy/PyTorch training dependencies required')
class CorpusPrefixTests(unittest.TestCase):
    def test_fixed_token_order_and_incomplete_utf8_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.zip'
            raw = "Title <text> Alpha beta can't gamma 中文".encode()
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('fixture', raw)
            text, digest = corpus_prefix(path, 'fixture', 5, len(raw) - 1)
            self.assertEqual(text, "title text alpha beta can't")
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, 'fewer'):
                corpus_prefix(path, 'fixture', 100, len(raw))

    def test_malformed_interior_bytes_are_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('fixture', b'alpha\xff beta gamma')
            with self.assertRaises(UnicodeDecodeError):
                corpus_prefix(path, 'fixture', 2, 100)


if __name__ == '__main__':
    unittest.main()
