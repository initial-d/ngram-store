"""Scheduling must not break paired evidence or duplicate existing workers."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import numpy
except ImportError:
    numpy = None

@unittest.skipUnless(numpy is not None, 'NumPy research dependency required')
class PairedBlockTest(unittest.TestCase):
    def test_rotations_keep_all_methods_and_batches_in_each_block(self):
        from tools.run_native_gpu_blocks import paired_cells
        from tools.tree_node_quality import write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); write(root / 'selection.json', {})
            entries = [{'codec': 'sq8', 'variant': v, 'job': f'{v}.json',
                **({} if v == 'direct' else {'block_rows': 16, 'predictor': 'xor'})}
                for v in ('direct', 'prior', 'zlib', 'libdeflate')]
            blocks = paired_cells({'entries': entries}, {'selected': [entries[-1]]}, root)
            self.assertEqual(len(blocks), 3)
            for i, block in enumerate(blocks):
                self.assertEqual(len(block['cells']), 8)
                self.assertEqual({(c['job'], c['batch_size']) for c in block['cells']},
                    {(e['job'], b) for e in entries for b in (1, 4)})
                self.assertEqual(block['cells'][0]['job'], entries[i]['job'])

    def test_partial_or_mixed_device_history_is_never_reused(self):
        from tools.run_native_gpu_blocks import reusable_block
        from tools.tree_node_quality import digest, write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job = root / 'job.json'; write(job, {})
            cells = [{'job': str(job), 'batch_size': b, 'output': str(root / f'{i}.json')}
                for i, b in enumerate((1, 4))]
            block = {'cells': cells}
            value = {'status': 'passed', 'measurement_host': 'host', 'gpu_before': {'gpu_uuid': 'GPU-a'},
                'cpu_affinity': [0], 'phase_source_sha256': 'code', 'job_sha256': digest(job)}
            write(cells[0]['output'], {**value, 'batch_size': 1})
            self.assertFalse(reusable_block(block))
            write(cells[1]['output'], {**value, 'batch_size': 4})
            self.assertTrue(reusable_block(block))
            for change in ({'gpu_before': {'gpu_uuid': 'GPU-b'}}, {'cpu_affinity': [1]},
                           {'gpu_samples': [{'external_compute_pids': [999]}]}, {'job_sha256': 'changed'}):
                write(cells[1]['output'], {**value, 'batch_size': 4, **change})
                self.assertFalse(reusable_block(block))

    def test_reservation_covers_cuda_context_gaps_and_expires(self):
        from tools.run_large_application_frontier import peer_reserved_gpus
        from tools.tree_node_quality import write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'native-parallel/dispatch.json', {'status': 'running',
                'controller_pid': 1234, 'reserved_gpus': list(range(8)), 'running': {}})
            with patch.object(Path, 'read_bytes', return_value=str(root).encode()):
                self.assertEqual(peer_reserved_gpus(root), set(range(8)))
            with patch.object(Path, 'read_bytes', side_effect=FileNotFoundError):
                self.assertEqual(peer_reserved_gpus(root), set())

    def test_adoption_observes_original_worker_and_final_artifact(self):
        from tools.run_large_application_frontier import AdoptedChild
        from tools.tree_node_quality import write
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / 'built'; argv = ['taskset', '-c', '0', 'python', '-m',
                'tools.large_application_frontier', 'build', '--output', str(result)]
            item = {'pid': 1234, 'command': argv}; child = AdoptedChild(item)
            with patch.object(Path, 'read_bytes', return_value=b'\0'.join(a.encode() for a in argv[3:])+b'\0'):
                self.assertIsNone(child.poll())
            write(result / 'built.json', {'status': 'complete'})
            with patch.object(Path, 'read_bytes', side_effect=FileNotFoundError):
                self.assertEqual(child.poll(), 0)
            self.assertFalse(item['adoption_terminal_evidence']['original_exit_code_observed'])

if __name__ == '__main__': unittest.main()
