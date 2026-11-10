"""Prompt sharding retains every official request and matched hardware."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.gpu_prompt_shards import aggregate,targets,digest
from tools.verify_memory_application import llm_timing


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


if __name__=='__main__':unittest.main()
