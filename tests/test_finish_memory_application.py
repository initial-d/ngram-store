"""A partial or failed serving study must not become a complete publication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.finish_memory_application import commits_in, sha, wait_for_serving


class PreparationGatesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.controller = self.root / 'final-serving-dispatch.json'

    def write(self, value):
        self.controller.write_text(json.dumps(value))

    def test_failed_controller_is_preserved_and_rejected(self):
        self.write({'status': 'failed', 'error': 'GPU competition'})
        before = self.controller.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Final serving failed'):
            wait_for_serving(self.root, 1, 1, lambda state: None)
        self.assertEqual(self.controller.read_bytes(), before)

    def test_complete_marker_requires_all_final_phases(self):
        phases = ['retime-llm-gpu', 'retime-llm-cpu', 'retime-long']
        for actual in (phases[:-1], phases + [phases[0]]):
            self.write({'status': 'complete', 'complete': [{'name': name} for name in actual], 'failures': []})
            with self.assertRaisesRegex(ValueError, 'Missing or repeated'):
                wait_for_serving(self.root, 1, 1, lambda state: None)
        self.write({'status': 'complete', 'complete': [{'name': name} for name in phases], 'failures': []})
        self.assertEqual(wait_for_serving(self.root, 1, 1, lambda state: None), sha(self.controller))
        self.write({'status': 'complete', 'complete': [{'name': name} for name in phases], 'failures': ['unresolved']})
        with self.assertRaisesRegex(ValueError, 'unresolved failures'):
            wait_for_serving(self.root, 1, 1, lambda state: None)

    def test_expired_wait_preserves_active_controller(self):
        self.write({'status': 'running-serial-final-serving', 'complete': []})
        before = self.controller.read_bytes(); changes = []
        with patch('tools.finish_memory_application.time.monotonic', side_effect=[0., 2.]):
            with self.assertRaises(TimeoutError):
                wait_for_serving(self.root, 1, 1, changes.append)
        self.assertEqual(changes, ['running-serial-final-serving'])
        self.assertEqual(self.controller.read_bytes(), before)

    def test_inventory_uses_commits_not_arbitrary_sha256_values(self):
        first, second = 'a' * 40, 'b' * 40
        value = {'environment': {'commit': first}, 'rows': [{'environment': {'code_sha': second}}],
                 'source_commit': 'unknown', 'checkpoint_sha256': 'c' * 64, 'commit': 'd' * 64}
        self.assertEqual(commits_in(value), {first, second})


if __name__ == '__main__':
    unittest.main()
