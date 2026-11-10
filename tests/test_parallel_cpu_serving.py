"""Parallel timing must retain all requests, pin physical cores and audit outputs."""
import hashlib
import importlib.util
import json
import os
import socket
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.parallel_cpu_serving import compare_outputs, digest, reference_requests, validate_plan
from tools.verify_memory_application import cpu_projection, verify_cpu_schedule


class CpuParallelTests(unittest.TestCase):
    def test_cli_consumes_reference_that_finished_before_process_start(self):
        from tools.parallel_cpu_serving import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'job.json';job.write_text('{}')
            plan=root/'cpu-serving-parallel-plan.json';plan.write_text('{}')
            registered=root/'cpu-parallel-reference-plan.json'
            registered.write_text(json.dumps({'measurement_host':socket.gethostname()}))
            output=root/'reference.json'
            output.write_text(json.dumps({'reference_execution':{'plan_sha256':digest(registered)}}))
            argv=['worker','--phase','retime-long','--job',str(job),'--output',str(output),
                  '--plan',str(plan),'--lane','0','--cohort','serial-reference']
            before=output.read_bytes()
            with patch('sys.argv',['worker',*argv]),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}), \
                 patch('tools.parallel_cpu_serving.worker') as numerical:main();numerical.assert_not_called()
            self.assertEqual(output.read_bytes(),before)
            argv[-1]='parallel-shared-host'
            with patch('sys.argv',['worker',*argv]),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}), \
                 self.assertRaises(SystemExit):main()

    def test_smt_siblings_are_not_independent_physical_lanes(self):
        plan = {'schema_version': 1, 'workers': 2, 'lanes': {
            '0': {'cpus': [0], 'physical_cores': [[0, 0]]},
            '1': {'cpus': [128], 'physical_cores': [[0, 0]]}},
            'model_lanes': {'retime-long': {'a': '0', 'b': '1'}}}
        with self.assertRaisesRegex(ValueError, 'physical CPU'):
            validate_plan(plan, 'retime-long', ['a', 'b'], {0, 128})

    def test_reference_scope_uses_frozen_primary_quality_choices(self):
        selected = {'models': [{'name': 'a', 'choices': [
            {'name': 'faiss-good', 'budget': .001, 'cached': 0},
            {'name': 'faiss-loose', 'budget': .02, 'cached': 0}]}]}
        requests = [{'model': 'a', 'name': name, 'cached': 0, 'index': index, 'cache': 0}
                    for name, index in (('raw-fp32', 'datrie'), ('raw-fp32', 'hash'),
                                        ('faiss-good', 'datrie'), ('faiss-loose', 'datrie'))]
        self.assertEqual(reference_requests(requests, selected, 'retime-long'), requests[:3])

    def test_reference_can_change_latency_but_never_vector_bytes(self):
        a = {'decoded_row_sha256': 'same', 'query': {'oov': {'output_sha256': 'same', 'complete_p50_ms': 3.}}}
        b = {'decoded_row_sha256': 'same', 'query': {'oov': {'output_sha256': 'same', 'complete_p50_ms': 1.}}}
        self.assertEqual(compare_outputs(a, b), {'oov': 3.})
        b['query']['oov']['output_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'complete query output'):
            compare_outputs(a, b)

    def test_public_verifier_rejects_missing_or_mismatched_serial_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root / 'llm/models/m0'; folder.mkdir(parents=True)
            plan = {'workers': 8, 'lanes': {str(i): {'cpus': [i], 'physical_cores': [[0, i]]} for i in range(8)},
                    'model_lanes': {'retime-llm-cpu': {'m0': '0'}}}
            (root / 'cpu-serving-parallel-plan.json').write_text(json.dumps(plan))
            plan_hash = digest(root / 'cpu-serving-parallel-plan.json')
            request = {'model': 'm0', 'name': 'raw-fp32', 'cache': 0, 'batch': 1, 'mode': 'codec', 'job': 'job'}
            worker_hash = digest(Path(__file__).parents[1] / 'tools/parallel_cpu_serving.py')
            a = {'decoded_row_sha256': 'rows', 'cpu_affinity': [0], 'cpu_lane': '0',
                 'cpu_parallel_plan_sha256': plan_hash, 'measurement_cohort': 'parallel-shared-host',
                 'worker_source_sha256': worker_hash, 'environment': {'commit': 'fixture'},
                 'query': {'test': {'input_hashes': ['input'], 'output_hashes': ['output'], 'p50_ms': 2.}}}
            b = {**a, 'measurement_cohort': 'serial-reference', 'query': {
                 'test': {'input_hashes': ['input'], 'output_hashes': ['output'], 'p50_ms': 1.}}}
            record = {'request': request, 'report': b, 'private_sha256': 'serial', 'parallel_to_serial_p50_ratios': {'test': 2.}}
            import gzip
            def save_reference():
                with gzip.open(folder / 'serial-references.json.gz', 'wt') as stream: json.dump([record], stream)
            save_reference()
            controller = {'status': 'complete', 'workers': 8, 'serial_reference_workers': 1,
                'cpu_parallel_plan_sha256': plan_hash, 'model_lanes': plan['model_lanes']['retime-llm-cpu'],
                'lanes': plan['lanes'], 'requests': [request], 'complete': [{}], 'reference_requests': [request],
                'serial_references': [{'request': request, 'sha256': 'serial', 'parallel_output_sha256': 'parallel',
                                       'parallel_to_serial_p50_ratios': {'test': 2.}}]}
            frozen = {'models': [{'name': 'm0', 'choices': []}]}
            records = [{'request': request, 'report': cpu_projection(a), 'private_sha256': 'parallel'}]
            inventory = {'fixture': {'files': {'tools/parallel_cpu_serving.py': {'original_sha256': worker_hash}}}}
            private = {'cpu-serving-parallel-plan.json': plan_hash}
            with patch('tools.verify_memory_application.source'), patch('tools.verify_memory_application.llm_timing'):
                verify_cpu_schedule(root, controller, frozen, records, 'retime-llm-cpu', 'llm/models', inventory, private)
                b['query']['test']['output_hashes'] = ['changed']; save_reference()
                with self.assertRaises(AssertionError):
                    verify_cpu_schedule(root, controller, frozen, records, 'retime-llm-cpu', 'llm/models', inventory, private)
                b['query']['test']['output_hashes'] = ['output']; save_reference()
                controller['serial_references'] = []
                with self.assertRaises(AssertionError):
                    verify_cpu_schedule(root, controller, frozen, records, 'retime-llm-cpu', 'llm/models', inventory, private)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Optional NumPy required for queue environment')
    def test_full_queue_parallelizes_then_runs_serial_references_and_resumes(self):
        from tools.parallel_cpu_serving import run_queue
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); frontier = root / 'frontier'; ready = root / 'ready.json'; plan_path = root / 'plan.json'
            write(ready, {'status': 'quality-complete'}); (root / 'all-compute-finished').touch()
            models = [f'm{i}' for i in range(8)]; requests = []; selected = []
            for model in models:
                selected.append({'name': model, 'choices': [
                    {'name': 'faiss-good', 'budget': .001, 'cached': 0},
                    {'name': 'faiss-loose', 'budget': .02, 'cached': 0}]})
                for name in ('raw-fp32', 'faiss-good', 'faiss-loose'):
                    job = root / model / (name + '.json'); write(job, {'fixture': True})
                    requests.append({'model': model, 'name': name, 'job': str(job), 'cached': 0, 'index': 'datrie', 'cache': 0})
            write(frontier / 'selection.json', {'models': selected})
            plan = {'schema_version': 1, 'workers': 8,
                    'lanes': {str(i): {'cpus': [i], 'physical_cores': [[0, i]]} for i in range(8)},
                    'model_lanes': {'retime-long': {name: str(i) for i, name in enumerate(models)}}}
            write(plan_path, plan)
            args = SimpleNamespace(phase='retime-long', ready=ready, frontier=frontier, output=root / 'timers', requests=32)
            barrier = threading.Barrier(8); active = set(); lock = threading.Lock(); seen = []; counts = []
            def fake(command, env, stdout, stderr):
                def arg(name): return command[command.index('--' + name) + 1]
                lane = arg('lane'); cohort = arg('cohort'); job = Path(arg('job')); output = Path(arg('output'))
                self.assertEqual(arg('cpu-list'), lane)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                with lock:
                    self.assertNotIn(lane, active); active.add(lane); counts.append(len(active)); seen.append(cohort)
                    if cohort == 'serial-reference': self.assertEqual(len(active), 1)
                if cohort == 'parallel-shared-host' and job.stem == 'raw-fp32': barrier.wait(timeout=5)
                write(output, {'status': 'complete', 'job_sha256': digest(job),
                    'cpu_parallel_plan_sha256': digest(plan_path), 'cpu_lane': lane, 'cpu_affinity': [int(lane)],
                    'measurement_cohort': cohort, 'decoded_row_sha256': 'rows',
                    'query': {'oov': {'output_sha256': 'vectors', 'complete_p50_ms': 2. if cohort == 'parallel-shared-host' else 1.}}})
                with lock: active.remove(lane)
                return SimpleNamespace(returncode=0)
            with patch('os.sched_getaffinity', return_value=set(range(8)), create=True), \
                 patch('tools.tree_node_quality.environment', return_value={'fixture': True}), \
                 patch('tools.parallel_cpu_serving.subprocess.run', side_effect=fake) as execution:
                run_queue(args, requests, {'status': 'quality-complete'}, plan_path)
                self.assertEqual(execution.call_count, 40)
                self.assertEqual(max(counts), 8)
                self.assertEqual(seen[:24], ['parallel-shared-host'] * 24)
                self.assertEqual(seen[24:], ['serial-reference'] * 16)
                self.assertEqual(json.loads((args.output / 'report.json').read_text())['status'], 'complete')
                execution.reset_mock()
                from tools.tree_node_quality import write as persist
                with patch('tools.parallel_cpu_serving.time.monotonic', return_value=1000.), \
                     patch('tools.tree_node_quality.write', wraps=persist) as checkpoints:
                    run_queue(args, requests, {'status': 'quality-complete'}, plan_path)
                    # Initial state, reference boundary and final state remain
                    # durable while retained receipts avoid per-item rewrites.
                    self.assertEqual(checkpoints.call_count, 3)
                saved=json.loads((args.output / 'report.json').read_text())
                self.assertEqual(len(saved['complete']), 24)
                self.assertEqual(len(saved['serial_references']), 16)
                execution.assert_not_called()
                output = args.output / '00000.json'; value = json.loads(output.read_text()); value['decoded_row_sha256'] = 'corrupt'; write(output, value)
                with self.assertRaisesRegex(RuntimeError, 'grid incomplete'):
                    run_queue(args, requests, {'status': 'quality-complete'}, plan_path)
