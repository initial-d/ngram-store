"""Independent model preparation preserves order, bytes and failure handling."""
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.evidence_parallel import ordered_map


@unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork required')
class EvidenceParallelTests(unittest.TestCase):
    def test_local_callbacks_execute_in_separate_processes_in_original_order(self):
        parent=os.getpid()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            result=list(ordered_map(lambda value:(value,os.getpid()),list(range(12)),4))
        self.assertEqual([v for v,pid in result],list(range(12)))
        self.assertTrue(all(pid!=parent for v,pid in result))
        self.assertGreaterEqual(len({pid for v,pid in result}),2)

    def test_complete_numeric_model_artifacts_are_identical_to_serial_execution(self):
        try:
            from tools.tree_node_quality import bootstrap, digest, write
        except ImportError:
            self.skipTest('optional NumPy required for numeric artifact comparison')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def worker(index,folder):
                values={k:[float(index+j+1) for j in range(8)] for k in
                        ('count','candidate_loss','baseline_loss','candidate_discrimination','baseline_discrimination')}
                # Real deterministic interval computation and canonical gzip writer.
                intervals=bootstrap(values,repeats=200,seed=42)
                path=root/folder/f'm{index}.json.gz';write(path,intervals)
                return digest(path)
            serial=list(ordered_map(lambda i:worker(i,'serial'),list(range(8)),1))
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                parallel=list(ordered_map(lambda i:worker(i,'parallel'),list(range(8)),4))
            self.assertEqual(serial,parallel)

    def test_worker_failure_is_preserved_and_does_not_poison_next_run(self):
        def fail(value):
            if value==2:raise ValueError('retained numerical failure')
            return value
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(ValueError,'retained numerical failure'):
                list(ordered_map(fail,list(range(5)),3))
            self.assertEqual(list(ordered_map(lambda i:i*i,list(range(4)),2)),[0,1,4,9])

    def test_gpu_parent_and_invalid_worker_counts_are_rejected(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaises(ValueError):list(ordered_map(lambda x:x,[1],2))
        for jobs in (0,65):
            with self.assertRaises(ValueError):list(ordered_map(lambda x:x,[1],jobs))


if __name__=='__main__':unittest.main()
