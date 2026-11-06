import json
from pathlib import Path
import tempfile
import unittest

from tools.counterfact_resume import completed_task, sha


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / 'prepared-qwen06-20877'
        self.prepared.mkdir()
        (self.prepared / 'batches.json').write_text(json.dumps([[0], [1], [2], [3], [4], [5], [6]]))
        self.meta = {'batch_size': 8, 'batches_sha256': sha(self.prepared / 'batches.json')}
        self.save(self.prepared / 'prepare.json', self.meta)

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_complete_cache_reuse_and_corruption(self):
        task = {'action': 'cache', 'model': 'qwen06', 'edits': 20877, 'shard': 0, 'name': 'cache0'}
        cache = self.root / 'cache-qwen06-20877'
        cache.mkdir()
        raw = cache / 'shard-0.f32'
        raw.write_bytes(b'\0' * 16)
        report = {'status': 'complete', 'shard': 0, 'shards': 6, 'batch_size': 8,
                  'prepared_sha256': sha(self.prepared / 'prepare.json'), 'bytes': 16,
                  'sha256': sha(raw), 'tokens': 4, 'hidden_size': 1,
                  'sequences': {'0': [0, 2], '6': [2, 2]}, 'frozen_weight_sha256': 'fixed'}
        self.save(cache / 'shard-0.json', report)
        self.save(self.root / 'cache-qwen06-1024/shard-0.json', {'frozen_weight_sha256': 'fixed'})
        result = completed_task(self.root, task, 'protocol')
        self.assertTrue(result['reused'])
        raw.write_bytes(b'1' * 16)
        with self.assertRaisesRegex(ValueError, 'cache bytes changed'):
            completed_task(self.root, task, 'protocol')

    def test_cache_coverage_failure_and_partial_not_overwritten(self):
        task = {'action': 'cache', 'model': 'qwen06', 'edits': 20877, 'shard': 0, 'name': 'cache0'}
        raw = self.root / 'cache-qwen06-20877/shard-0.f32'
        raw.parent.mkdir()
        raw.write_bytes(b'\0' * 16)
        with self.assertRaisesRegex(ValueError, 'partial cache'):
            completed_task(self.root, task, 'protocol')
        report = {'status': 'complete', 'shard': 0, 'shards': 6, 'batch_size': 8,
                  'prepared_sha256': sha(self.prepared / 'prepare.json'), 'bytes': 16,
                  'sha256': sha(raw), 'tokens': 4, 'hidden_size': 1,
                  'sequences': {'0': [0, 2], '99': [2, 2]}, 'frozen_weight_sha256': 'fixed'}
        self.save(raw.with_suffix('.json'), report)
        with self.assertRaisesRegex(ValueError, 'sequence coverage changed'):
            completed_task(self.root, task, 'protocol')
        self.assertEqual(raw.read_bytes(), b'\0' * 16)

    def test_training_delta_and_protocol_verified(self):
        task = {'action': 'train', 'model': 'qwen06', 'edits': 20877, 'seed': 3,
                'variant': 'large', 'name': 'qwen06-20877-large-s3'}
        folder = self.root / 'runs' / task['name']
        folder.mkdir(parents=True)
        (folder / 'delta.safetensors').write_bytes(b'fixture-delta')
        self.save(folder / 'case-scores.json', [])
        report = {'status': 'complete', 'protocol_sha256': 'protocol',
                  'prepared_sha256': sha(self.prepared / 'prepare.json'), 'seed': 3, 'variant': 'large',
                  'case_scores_sha256': sha(folder / 'case-scores.json'),
                  'delta': {'bytes': 13, 'sha256': sha(folder / 'delta.safetensors')}}
        self.save(folder / 'report.json', report)
        self.assertTrue(completed_task(self.root, task, 'protocol')['reused'])
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            completed_task(self.root, task, 'other-protocol')
        (folder / 'delta.safetensors').write_bytes(b'broken--delta')
        with self.assertRaisesRegex(ValueError, 'training delta changed'):
            completed_task(self.root, task, 'protocol')


if __name__ == '__main__':
    unittest.main()
