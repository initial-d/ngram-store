"""Advance complete CPU work without changing hardware or overlapping core users."""
import json
import importlib.util
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.cpu_phase_overlap import ready, run, validate
from tools.parallel_cpu_serving import digest


class CpuPhaseOverlapTests(unittest.TestCase):
    def fixture(self, root):
        requests=[]
        lanes={str(i):{'cpus':[i]} for i in range(6)}
        for i in range(6):
            job=root/f'm{i}.json';job.write_text('{}')
            for cache in (0,1048576):
                requests.append({'model':f'm{i}','name':'native','job':str(job),
                                 'mode':'codec','batch':1,'cache':cache})
        hardware={'lanes':lanes,'model_lanes':{'retime-llm-cpu':{f'm{i}':str(i) for i in range(6)}}}
        original=root/'cpu-serving-parallel-plan.json';original.write_text(json.dumps(hardware))
        plan={'measurement_host':socket.gethostname(),'original_cpu_plan_sha256':digest(original),
              'requests':requests,'upstream':str(root)}
        (root/'cpu-phase-overlap-plan.json').write_text(json.dumps(plan))
        (root/'retime-long').mkdir()
        (root/'retime-long/report.json').write_text(json.dumps({'status':'running-serial-references',
                                                              'requests':[{}],'complete':[{}]}))
        return requests,original

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy required')
    def test_complete_six_lane_cohort_uses_original_fresh_worker_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);requests,original=self.fixture(root)
            barrier=threading.Barrier(6);seen=set();lock=threading.Lock()
            def execute(command,stdout,stderr,env):
                arg=lambda key:command[command.index('--'+key)+1]
                lane=arg('lane')
                with lock:first=lane not in seen;seen.add(lane)
                if first:barrier.wait(timeout=5)
                self.assertEqual(arg('cpu-list'),lane)
                self.assertEqual(arg('plan'),str(original))
                self.assertEqual(arg('requests'),'32')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
                output=Path(arg('output'));self.assertFalse(output.exists())
                output.write_text(json.dumps({'status':'complete','job_sha256':digest(arg('job')),
                    'cpu_parallel_plan_sha256':digest(original),'cpu_lane':lane,'cpu_affinity':[int(lane)],
                    'measurement_cohort':'parallel-shared-host'}))
                return SimpleNamespace(returncode=0)
            with patch('tools.finalize_memory_application.llm_requests',return_value=requests), \
                 patch('tools.tree_node_quality.environment',return_value={'fixture':True}), \
                 patch('tools.cpu_phase_overlap.subprocess.run',side_effect=execute):run(root)
            report=json.loads((root/'cpu-phase-overlap/report.json').read_text())
            self.assertEqual(report['status'],'complete');self.assertEqual(len(report['complete']),12)
            self.assertEqual(seen,set(map(str,range(6))))
            self.assertEqual({json.dumps(v['request'],sort_keys=True) for v in report['complete']},
                             {json.dumps(v,sort_keys=True) for v in requests})
            self.assertFalse((root/'retime-llm-cpu/report.json').exists())

    def test_gate_requires_all_sgns_main_outputs_and_rejects_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.assertFalse(ready(root));self.fixture(root)
            self.assertTrue(ready(root))
            target=root/'retime-long/report.json'
            target.write_text(json.dumps({'status':'running','requests':[{},{}],'complete':[{}]}))
            self.assertFalse(ready(root))
            target.write_text(json.dumps({'status':'failed'}))
            with self.assertRaises(RuntimeError):ready(root)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy required')
    def test_changed_plan_or_incomplete_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);requests,original=self.fixture(root)
            with patch('tools.finalize_memory_application.llm_requests',return_value=requests):validate(root)
            with patch('tools.finalize_memory_application.llm_requests',return_value=requests[:-1]):
                with self.assertRaises(ValueError):validate(root)
            original.write_text('{}')
            with self.assertRaises(ValueError):validate(root)


if __name__=='__main__':unittest.main()
