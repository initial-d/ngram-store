"""Reference concurrency preserves full coverage, fixed cores and output hashes."""
import json
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.cpu_parallel_references import run,wait_for_reference
from tools.parallel_cpu_serving import digest
from tools.verify_memory_application import cpu_projection,verify_cpu_schedule


class ParallelReferenceTests(unittest.TestCase):
    def test_all_reference_requests_run_on_eight_original_core_lanes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);phase='retime-long';hardware={'lanes':{str(i):{'cpus':[i]} for i in range(8)},
                'model_lanes':{phase:{f'm{i}':str(i) for i in range(8)}},'local_gpu_finished_marker':'gpu-done.json'}
            cpu_plan=root/'cpu-serving-parallel-plan.json';cpu_plan.write_text(json.dumps(hardware))
            requests=[]
            for i in range(8):
                for name in ('raw-fp32','raw-fp16','raw-int8'):
                    job=root/f'm{i}-{name}.json';job.write_text('{}')
                    requests.append({'model':f'm{i}','name':name,'job':str(job),'cache':0,'cached':0,'index':'datrie'})
            original=root/phase;original.mkdir()
            (original/'report.json').write_text(json.dumps({'status':'running-serial-references','requests':requests,'complete':requests}))
            (root/'gpu-done.json').write_text(json.dumps({'status':'complete','measurement_host':socket.gethostname()}))
            (root/'long-frontier').mkdir();(root/'long-frontier/selection.json').write_text(json.dumps({
                'models':[{'name':f'm{i}','choices':[]} for i in range(8)]}))
            plan={'measurement_host':socket.gethostname(),'original_cpu_plan_sha256':digest(cpu_plan),
                'phases':{phase:{'requests':requests}},'workers':8,'upstream':str(root)}
            path=root/'cpu-parallel-reference-plan.json';path.write_text(json.dumps(plan))
            barrier=threading.Barrier(8);seen=set();lock=threading.Lock()
            def execute(command,env,stdout,stderr):
                arg=lambda key:command[command.index('--'+key)+1]
                lane=arg('lane')
                with lock:first=lane not in seen;seen.add(lane)
                if first:barrier.wait(timeout=5)
                self.assertEqual(arg('cpu-list'),lane);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
                output=Path(arg('output'));output.write_text(json.dumps({'status':'complete','job_sha256':digest(Path(arg('job'))),
                    'cpu_parallel_plan_sha256':digest(cpu_plan),'cpu_lane':lane,'cpu_affinity':[int(lane)],
                    'measurement_cohort':'serial-reference','reference_execution':{'plan_sha256':digest(path)}}))
                return SimpleNamespace(returncode=0)
            with patch('tools.tree_node_quality.environment',return_value={'fixture':True}), \
                 patch('tools.cpu_parallel_references.subprocess.run',side_effect=execute):run(root,phase)
            report=json.loads((root/'cpu-parallel-references'/phase/'report.json').read_text())
            self.assertEqual(report['status'],'complete');self.assertEqual(len(report['complete']),24)
            self.assertEqual(seen,set(map(str,range(8))))
            self.assertEqual({json.dumps(r['request'],sort_keys=True) for r in report['complete']},
                             {json.dumps(r,sort_keys=True) for r in requests})

    def test_dispatcher_wait_returns_only_an_actual_registered_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'cpu-parallel-reference-plan.json'
            plan.write_text(json.dumps({'measurement_host':socket.gethostname()}))
            output=root/'reference.json';output.write_text(json.dumps({'reference_execution':{'plan_sha256':digest(plan)}}))
            args=SimpleNamespace(plan=root/'cpu-serving-parallel-plan.json',phase='retime-long',cohort='serial-reference',output=output)
            self.assertTrue(wait_for_reference(args))
            output.write_text(json.dumps({'reference_execution':{'plan_sha256':'changed'}}))
            with self.assertRaises(ValueError):wait_for_reference(args)
            args.cohort='parallel-shared-host';self.assertFalse(wait_for_reference(args))

    def test_public_verifier_checks_actual_reference_execution_and_source(self):
        import gzip
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'llm/models/m0';folder.mkdir(parents=True)
            phase='retime-llm-cpu';plan={'workers':8,'lanes':{
                str(i):{'cpus':[i],'physical_cores':[[0,i]]} for i in range(8)},'model_lanes':{phase:{'m0':'0'}}}
            cpu=root/'cpu-serving-parallel-plan.json';cpu.write_text(json.dumps(plan));cpu_hash=digest(cpu)
            ref=root/'cpu-parallel-reference-plan.json';ref.write_text(json.dumps({'workers':8,'original_cpu_plan_sha256':cpu_hash}));ref_hash=digest(ref)
            worker_hash=digest(Path(__file__).parents[1]/'tools/parallel_cpu_serving.py')
            source_hash=digest(Path(__file__).parents[1]/'tools/cpu_parallel_references.py')
            request={'model':'m0','name':'raw-fp32','cache':0,'batch':1,'mode':'codec','job':'job'}
            a={'decoded_row_sha256':'rows','cpu_affinity':[0],'cpu_lane':'0','cpu_parallel_plan_sha256':cpu_hash,
                'measurement_cohort':'parallel-shared-host','worker_source_sha256':worker_hash,'environment':{'commit':'fixture'},
                'query':{'test':{'input_hashes':['input'],'output_hashes':['output'],'p50_ms':2.}}}
            b={**a,'measurement_cohort':'serial-reference','query':{'test':{'input_hashes':['input'],'output_hashes':['output'],'p50_ms':1.}},
                'reference_execution':{'kind':'parallel-fixed-core','workers':8,'plan_sha256':ref_hash,'source_sha256':source_hash}}
            def save():
                with gzip.open(folder/'serial-references.json.gz','wt') as stream:json.dump([
                    {'request':request,'report':b,'private_sha256':'ref','parallel_to_serial_p50_ratios':{'test':2.}}],stream)
            controller={'status':'complete','workers':8,'serial_reference_workers':8,'reference_execution':'parallel-fixed-core',
                'parallel_reference_plan_sha256':ref_hash,'cpu_parallel_plan_sha256':cpu_hash,'model_lanes':plan['model_lanes'][phase],
                'lanes':plan['lanes'],'requests':[request],'complete':[{}],'reference_requests':[request],
                'serial_references':[{'request':request,'sha256':'ref','parallel_output_sha256':'primary','parallel_to_serial_p50_ratios':{'test':2.}}]}
            records=[{'request':request,'report':cpu_projection(a),'private_sha256':'primary'}]
            inventory={'fixture':{'files':{'tools/parallel_cpu_serving.py':{'original_sha256':worker_hash},
                                         'tools/cpu_parallel_references.py':{'original_sha256':source_hash}}}}
            private={'cpu-serving-parallel-plan.json':cpu_hash,'cpu-parallel-reference-plan.json':ref_hash}
            with patch('tools.verify_memory_application.source'),patch('tools.verify_memory_application.llm_timing'):
                save();verify_cpu_schedule(root,controller,{'models':[{'name':'m0','choices':[]}]},records,phase,'llm/models',inventory,private)
                for field,value in [('workers',1),('source_sha256','changed'),('plan_sha256','changed')]:
                    original=b['reference_execution'][field];b['reference_execution'][field]=value;save()
                    with self.subTest(field=field),self.assertRaises(AssertionError):
                        verify_cpu_schedule(root,controller,{'models':[{'name':'m0','choices':[]}]},records,phase,'llm/models',inventory,private)
                    b['reference_execution'][field]=original


if __name__=='__main__':unittest.main()
