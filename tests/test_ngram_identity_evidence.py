import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

EVIDENCE = Path(__file__).resolve().parents[1] / 'evidence/ngram_identity_evidence_20261107'
SPEC = importlib.util.spec_from_file_location('identity_evidence_verifier', EVIDENCE / 'verify.py')
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class IdentityEvidenceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'evidence'
        shutil.copytree(EVIDENCE, self.root, ignore=shutil.ignore_patterns('__pycache__'))

    def rewrite(self, path, data):
        path.write_text(json.dumps(data, indent=2) + '\n')
        manifest = json.loads((self.root / 'manifest.json').read_text())
        manifest['files'][str(path.relative_to(self.root))] = {
            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))

    def test_complete_archive(self):
        self.assertEqual(VERIFIER.verify(self.root), (24, 8))

    def test_changed_or_missing_case_array_is_rejected(self):
        arrays = self.root / 'paired-test-arrays.npz'
        arrays.write_bytes(b'corrupted paired case arrays')
        with self.assertRaisesRegex(AssertionError, 'paired-test-arrays'):
            VERIFIER.verify(self.root)
        arrays.unlink()
        with self.assertRaisesRegex(AssertionError, 'file set'):
            VERIFIER.verify(self.root)

    def test_unmatched_budget_rejected_even_with_refreshed_hashes(self):
        name = 'qwen06-1024-exact-s3'
        path = self.root / f'reports/runs/{name}/report.json'
        report = json.loads(path.read_text())
        report['trainable_parameters'] += 32
        report['memory_parameters'] += 32
        self.rewrite(path, report)
        analysis_path = self.root / 'final-analysis.json'
        analysis = json.loads(analysis_path.read_text())
        analysis['input_sha256'][f'runs/{name}/report.json'] = VERIFIER.digest(path)
        self.rewrite(analysis_path, analysis)
        with self.assertRaisesRegex(AssertionError, 'trainable_parameters'):
            VERIFIER.verify(self.root)

    def test_positive_claim_cannot_hide_uncertain_interval(self):
        path = self.root / 'final-analysis.json'
        analysis = json.loads(path.read_text())
        comparison = next(c for c in analysis['exact_vs_control']
                          if c['model'] == 'qwen06' and c['edits'] == 1024
                          and c['control'] == 'hash-native-oov')
        self.assertLess(comparison['harmonic_delta_interval95'][0], 0)
        comparison['gate_positive_H_with_NS_point_within_2pp'] = True
        self.rewrite(path, analysis)
        with self.assertRaises(AssertionError):
            VERIFIER.verify(self.root)


if __name__ == '__main__':
    unittest.main()
