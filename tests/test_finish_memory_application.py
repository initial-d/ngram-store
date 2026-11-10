"""A partial or failed serving study must not become a complete publication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.finish_memory_application import commits_in, sha, source_commits, wait_for_serving


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

    def test_inventory_includes_actual_gpu_children_and_serial_cpu_sources(self):
        def write(name,value):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        write('quality-ready.json',{})
        for name in ('long-row-audit/report.json','llm-row-audit/report.json'):
            write(name,{})
        for family in ('long-frontier','llm-frontier'):
            write(family+'/prepare.json',{'models':[]})
        for phase in ('retime-long','retime-llm-cpu','retime-llm-gpu'):
            write(phase+'/report.json',{})
        commits=['a'*40,'b'*40,'c'*40,'d'*40,'e'*40]
        write('retime-llm-gpu/00000.json',{'measurement_purpose':'routing-receipt-only'})
        write('gpu-prompt-shards/outputs/00000.json',
              {'shards':[{'report':{'environment':{'commit':commits[0]}}}]})
        write('gpu-spare-lanes/outputs/00000.json',{'environment':{'commit':commits[1]}})
        write('gpu-prompt-shards/report.json',
              {'preflights':[{'records':[{'report':{'environment':{'commit':commits[2]}}}]}]})
        write('retime-long/serial-reference/00000.json',{'environment':{'commit':commits[3]}})
        with patch('tools.finish_memory_application.subprocess.check_output',return_value=commits[4]), \
             patch('tools.finish_memory_application.subprocess.run') as validate:
            self.assertEqual(source_commits(self.root,self.root/'sources'),commits)
            self.assertEqual([call.args[0] for call in validate.call_args_list],
                             [['git','cat-file','-e',commit+'^{commit}'] for commit in commits])


if __name__ == '__main__':
    unittest.main()
