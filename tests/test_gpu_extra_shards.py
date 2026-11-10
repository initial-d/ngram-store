"""Two-replica cohorts retain original work and explicit routing provenance."""
import copy
import gzip
import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_gpu_prompt_shards import publication_fixture
from tools.gpu_extra_shards import aggregate, digest, route_pending_shard, targets
from tools.gpu_spare_lanes import check_result
from tools.verify_memory_application import llm_timing, verify_extra_shards


def fixture(root):
    controller,records,inventory,private=publication_fixture(root)
    old_plan=json.loads((root/'gpu-prompt-shard-plan.json').read_text())
    groups={model:[{'gpu':old[0]['gpu']+i,'cpus':old[i]['cpus'],'slice':[16*i,16]}
                   for i in range(2)] for model,old in old_plan['groups'].items()}
    requests=controller['requests']
    for request in requests:request['batch']=1
    for record in records:
        request=record['request'];group=groups[request['model']];old=record['report']['shards'];children=[]
        for i,lane in enumerate(group):
            value=copy.deepcopy(old[2*i]['report'])
            value.update(request_slice=lane['slice'],selected_gpu_selector=str(lane['gpu']),
                         cpu_affinity=lane['cpus'],batch_size=1)
            value['requests']=sum([copy.deepcopy(old[2*i+j]['report']['requests']) for j in range(2)],[])
            for query in value['requests']:
                query['prompt_hashes']=query['prompt_hashes'][:1];query['batch_case_ids']=[0]
            value['samples']=sum([old[2*i+j]['report']['samples'] for j in range(2)],[])
            value['unique_prompt_count']=16;value['decode_tokens_per_second']=500.
            value['gpu_before']['gpu_uuid']=f'GPU-{lane["gpu"]}'
            children.append({'report':value,'sha256':'child'})
        value=aggregate(request,children,group)
        value.update(extra_shard_plan_sha256='plan-hash',aggregation_source_sha256='gpu_extra_shards-hash')
        record['report']=value
    inventory['commit']['files']['tools/gpu_extra_shards.py']={'original_sha256':'gpu_extra_shards-hash'}
    controller.update(extra_shard_plan_sha256='plan-hash',extra_shard_groups=groups,extra_shard_host='host-b')
    sharded=json.loads((root/'llm/prompt-shard-controller.json').read_text())
    sharded['workers']=4;sharded['requests']=requests;sharded['complete']=controller['complete']
    preflights=[]
    for model,group in groups.items():
        for lane in group:
            entry=next(v for v in sharded['preflights'] if v['model']==model and v['gpu']==lane['gpu'])
            for child in entry['records']:
                child['report']['request_slice']=[lane['slice'][0],2]
            preflights.append(entry)
    sharded['preflights']=preflights
    parent={'requests':requests,'complete':[{'request':r,'sha256':f'prior-{i}'} for i,r in enumerate(requests)]}
    prior=[{'request':r,'private_sha256':f'prior-{i}','report':{
        'measurement_purpose':'extra-shard-routing-receipt-only','request':r,
        'extra_shard_plan_sha256':'plan-hash','shard_measurement_sha256':f'composite-{i}'}}
        for i,r in enumerate(requests)]
    for name,value in (('gpu-extra-shard-plan.json',{'schema_version':1,'groups':groups,'requests':requests,'measurement_host':'host-b'}),
                       ('llm/extra-shard-controller.json',sharded),('llm/pre-extra-sharded-gpu-controller.json',parent)):
        (root/name).write_text(json.dumps(value))
    with gzip.open(root/'llm/pre-extra-sharded-gpu-records.json.gz','wt') as stream:json.dump(prior,stream)
    return controller,records,inventory,{'gpu-extra-shard-plan.json':'plan-hash'}


class ExtraShardTests(unittest.TestCase):
    def test_full_batch1_cohorts_keep_32_requests_and_96_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=fixture(root)
            parent,prior=verify_extra_shards(root,*args)
            self.assertEqual(len(parent['requests']),4);self.assertEqual(len(prior),4)
            self.assertEqual(len(targets(json.loads((root/'gpu-extra-shard-plan.json').read_text()))),4)
            for record in args[1]:llm_timing(record['report'],'gpu')
            args[1][0]['report']['shards'][1]['report']['request_slice']=[8,16]
            with self.assertRaises(AssertionError):verify_extra_shards(root,*args)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy required')
    def test_routing_is_compatible_with_old_drains_but_has_no_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'canonical').mkdir();(root/'gpu-extra-shards/outputs').mkdir(parents=True)
            job=root/'job.json';job.write_text('{}')
            request={'model':'m','job':str(job),'mode':'codec','batch':1,'cache':0,'name':'native'}
            plan={'canonical_directory':str(root/'canonical'),'requests':[request], 'groups':{'m':[]},
                  'original_cpu_affinity':{'m':[3]},'original_devices':{'m':3}}
            (root/'gpu-extra-shard-plan.json').write_text(json.dumps(plan))
            (root/'gpu-extra-shards/outputs/00000.json').write_text(json.dumps({'status':'passed','job_sha256':digest(job)}))
            args=SimpleNamespace(output=root/'canonical/00000.json',job=job,mode='codec',cache_bytes=0,
                batch_size=1,requests=32,repeat=3,warmup=1,decode_steps=16)
            with patch('os.sched_getaffinity',return_value={3},create=True):self.assertTrue(route_pending_shard(args))
            receipt=json.loads(args.output.read_text());check_result(receipt,request,3,[3])
            self.assertTrue(receipt['routing_configuration_only'])
            self.assertNotIn('samples',receipt);self.assertNotIn('open_seconds',receipt)
            self.assertIn('not a physical GPU snapshot',receipt['gpu_before']['scope'])


if __name__=='__main__':unittest.main()
