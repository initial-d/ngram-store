"""Prompt sharding retains every official request and matched hardware."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.gpu_prompt_shards import aggregate,targets,digest
from tools.verify_memory_application import llm_timing, verify_prompt_shards


def fixture(root):
    job=root/'job.json';job.write_text('{}')
    request={'job':str(job),'model':'m','mode':'codec','batch':8,'cache':0,'name':'native-none-fp32'}
    group=[{'gpu':i,'cpus':[i,i+128],'slice':[8*i,8]} for i in range(4)]
    records=[]
    for lane in group:
        start,count=lane['slice'];gpu=lane['gpu']
        queries=[{'prompt_hashes':[f'prompt-{i}-{j}' for j in range(8)],'batch_case_ids':list(range(8))} for i in range(start,start+count)]
        samples=[{'prefill_ms':float(i+1),'decode_ms':[float(i+2)]*16} for i in range(24)]
        report={'status':'passed','job_sha256':digest(job),'mode':'codec','batch_size':8,'cache_bytes':0,'decode_steps':16,
            'selected_gpu_selector':str(gpu),'cpu_affinity':lane['cpus'],'request_slice':lane['slice'],
            'requests':queries,'samples':samples,'unique_prompt_count':64,'open_seconds':1.,
            'gpu_before':{'compute_pids':[],'external_compute_pids':[],'utilization_percent':0},
            'gpu_samples':[{'compute_pids':[123],'external_compute_pids':[]}],
            'batch_contract':'Each batch uses distinct real official prompts',
            'prefill_p50_ms':12.5,'prefill_p95_ms':22.85,'decode_p50_ms':13.5,'decode_p95_ms':24.,
            'decode_tokens_per_second':8*1000/13.5}
        # The same per-request decode samples appear 16 times, so p95 uses
        # the repeated sample distribution rather than 24 scalar observations.
        report['decode_p95_ms']=24.
        records.append({'report':report,'sha256':'hash'})
    return request,group,records


class PromptShardTests(unittest.TestCase):
    def test_all_256_prompts_and_96_samples_survive_and_metrics_are_recomputed(self):
        with tempfile.TemporaryDirectory() as temp:
            request,group,records=fixture(Path(temp));value=aggregate(request,records,group)
            self.assertEqual(value['unique_prompt_count'],256);self.assertEqual(len(value['requests']),32)
            self.assertEqual(len(value['samples']),96);self.assertEqual(value['selected_gpu_selector'],'matched-prompt-shards')
            self.assertNotIn('gpu_before',value)
            llm_timing(value,'gpu')

    def test_missing_overlapping_or_shortened_slices_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            request,group,records=fixture(Path(temp))
            with self.assertRaisesRegex(ValueError,'Every'):aggregate(request,records[:3],group)
            altered=copy.deepcopy(records);altered[1]['report']['request_slice']=[4,8]
            with self.assertRaisesRegex(ValueError,'Overlapping'):aggregate(request,altered,group)
            altered=copy.deepcopy(records);altered[1]['report']['samples'].pop()
            with self.assertRaisesRegex(ValueError,'Three repeats'):aggregate(request,altered,group)

    def test_same_prompt_slice_cannot_silently_switch_device_or_model_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            request,group,records=fixture(Path(temp));records[1]['report']['selected_gpu_selector']='3'
            with self.assertRaisesRegex(ValueError,'hardware'):aggregate(request,records,group)
            records[1]['report']['selected_gpu_selector']='1';records[1]['report']['decoded_row_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'bytes'):aggregate(request,records,group)

    def test_plan_requires_disjoint_cards_and_cpu_cores(self):
        with tempfile.TemporaryDirectory() as temp:
            request,group,_=fixture(Path(temp));plan={'schema_version':1,'groups':{'m':group},'requests':[request]}
            self.assertEqual(targets(plan),[(0,request)])
            group[1]['cpus']=[0]
            with self.assertRaisesRegex(ValueError,'Disjoint'):targets(plan)


def publication_fixture(root):
    """Two complete model cohorts, eight devices, and prior routing receipts."""
    job=root/'job.json';job.write_text('{}')
    groups={model:[{'gpu':base+i,'cpus':[base+i,base+i+128],'slice':[8*i,8]}
                   for i in range(4)] for model,base in (('small',0),('large',4))}
    requests=[{'job':str(job),'model':model,'name':name,'mode':mode,'batch':8,'cache':0}
              for model in groups for name,mode in (('source-gpu','source'),('native-none-fp32','codec'))]
    plan={'schema_version':1,'groups':groups,'requests':requests,'measurement_host':'host-b'}
    files={f'tools/{name}.py':{'original_sha256':f'{name}-hash'} for name in
           ('tree_node_quality','gpu_prompt_shards','benchmark_llm_codec_serving',
            'cached_memory_decode','codec_memory_adapter')}
    environment={'commit':'commit','source_sha256':'tree_node_quality-hash',
                 'python':'3.12','torch':'2.8','os':'Linux','command':'fixture'}

    def child(request,lane,preflight=False):
        start,count=lane['slice'];count=2 if preflight else count
        queries=[{'prompt_hashes':[f'{request["model"]}-prompt-{i}-{j}' for j in range(8)],
                  'batch_case_ids':list(range(8)),'input_ids_sha256':f'input-{i}',
                  'cached_logit_hashes':[f'logit-{i}'],'generated_ids_sha256':f'ids-{i}'}
                 for i in range(start,start+count)]
        value={'status':'passed','environment':environment,'job_sha256':digest(job),
               'measurement_host':'host-b','mode':request['mode'],'batch_size':8,'cache_bytes':0,
               'decode_steps':16,'selected_gpu_selector':str(lane['gpu']),'cpu_affinity':lane['cpus'],
               'request_slice':[start,count],'requests':queries,'unique_prompt_count':count*8,
               'samples':[{'prefill_ms':1.,'decode_ms':[2.]*16} for _ in range(count*(1 if preflight else 3))],
               'open_seconds':1.,'prefill_p50_ms':1.,'prefill_p95_ms':1.,'decode_p50_ms':2.,
               'decode_p95_ms':2.,'decode_tokens_per_second':4000.,
               'batch_contract':'Each batch uses distinct real official prompts',
               'gpu_before':{'gpu_uuid':f'GPU-{lane["gpu"]}','compute_pids':[],
                             'external_compute_pids':[],'utilization_percent':0},
               'gpu_samples':[{'compute_pids':[123],'external_compute_pids':[]}],
               'phase_source_sha256':'benchmark_llm_codec_serving-hash',
               'cached_decode_source_sha256':'cached_memory_decode-hash',
               'codec_adapter_source_sha256':'codec_memory_adapter-hash'}
        return {'report':value,'sha256':'child-hash'}

    published=[];complete=[];prior=[];diagnostics=[]
    for i,request in enumerate(requests):
        group=groups[request['model']]
        report=aggregate(request,[child(request,lane) for lane in group],group)
        report.update(prompt_shard_plan_sha256='plan-hash',aggregation_source_sha256='gpu_prompt_shards-hash')
        published.append({'request':request,'private_sha256':f'composite-{i}','report':report})
        complete.append({'request':request,'sha256':f'composite-{i}'})
        prior.append({'request':request,'sha256':f'prior-{i}'})
        receipt={'measurement_purpose':'prompt-shard-routing-receipt-only',
                 'request':request,'shard_measurement_sha256':f'composite-{i}',
                 'prompt_shard_plan_sha256':'plan-hash'}
        diagnostics.append({'request':request,'private_sha256':f'prior-{i}','report':receipt})
    preflights=[]
    for model,group in groups.items():
        source,native=[r for r in requests if r['model']==model]
        for lane in group:
            preflights.append({'model':model,'gpu':lane['gpu'],
                               'records':[child(source,lane,True),child(native,lane,True)]})
    controller={'requests':requests,'complete':complete,'prompt_shard_plan_sha256':'plan-hash',
                'prompt_shard_groups':groups,'prompt_shard_host':'host-b'}
    sharded={'status':'complete','workers':8,'plan_sha256':'plan-hash',
             'requests':requests,'complete':complete,'preflights':preflights}
    parent={'requests':requests,'complete':prior}
    (root/'llm').mkdir()
    for name,value in (('gpu-prompt-shard-plan.json',plan),('llm/prompt-shard-controller.json',sharded),
                       ('llm/pre-sharded-gpu-controller.json',parent)):
        (root/name).write_text(json.dumps(value))
    with gzip.open(root/'llm/pre-sharded-gpu-records.json.gz','wt') as stream:json.dump(diagnostics,stream)
    return controller,published,{'commit':{'files':files}},{'gpu-prompt-shard-plan.json':'plan-hash'}


class PromptShardPublicationTests(unittest.TestCase):
    def test_complete_provenance_and_prior_receipts_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=publication_fixture(root)
            parent,prior=verify_prompt_shards(root,*args)
            self.assertEqual(len(parent['requests']),4)
            self.assertEqual(len(prior),4)
            for record in args[1]:llm_timing(record['report'],'gpu')

    def test_published_cohort_cannot_lose_or_duplicate_a_method(self):
        for change in ('missing','duplicate'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);args=publication_fixture(root)
                if change=='missing':args[1].pop()
                else:args[1][1]=copy.deepcopy(args[1][0])
                with self.assertRaises(AssertionError):verify_prompt_shards(root,*args)

    def test_device_host_input_and_source_changes_are_rejected(self):
        changes=(('measurement_host','other-host'),('cpu_affinity',[99]),
                 ('phase_source_sha256','changed-source'))
        for field,value in changes:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);args=publication_fixture(root)
                args[1][1]['report']['shards'][0]['report'][field]=value
                with self.assertRaises(AssertionError):verify_prompt_shards(root,*args)
        for field in ('gpu_uuid','input_ids_sha256'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);args=publication_fixture(root)
                child=args[1][1]['report']['shards'][0]['report']
                if field=='gpu_uuid':child['gpu_before'][field]='different-device'
                else:child['requests'][0][field]='different-prompt'
                with self.assertRaises(AssertionError):verify_prompt_shards(root,*args)

    def test_failed_preflight_or_wrong_prior_receipt_cannot_enter_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=publication_fixture(root)
            path=root/'llm/prompt-shard-controller.json';value=json.loads(path.read_text())
            value['preflights'][0]['records'][1]['report']['requests'][0]['cached_logit_hashes']=['changed']
            path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):verify_prompt_shards(root,*args)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=publication_fixture(root)
            path=root/'llm/pre-sharded-gpu-records.json.gz'
            with gzip.open(path,'rt') as stream:value=json.load(stream)
            value[0]['report']['shard_measurement_sha256']='unrelated-output'
            with gzip.open(path,'wt') as stream:json.dump(value,stream)
            with self.assertRaises(AssertionError):verify_prompt_shards(root,*args)


if __name__=='__main__':unittest.main()
