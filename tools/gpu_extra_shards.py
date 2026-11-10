"""Fill original-host spare GPUs with complete matched two-replica batch-1 cohorts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time

from .gpu_spare_lanes import command, digest, load, check_result
from .gpu_job_retry import run_gpu_job, wait_idle


def targets(plan):
    groups=plan['groups'];seen=set();cpus=set()
    if plan['schema_version']!=1:raise ValueError('Registered prompt-shard schema required')
    for group in groups.values():
        if len(group)!=2:raise ValueError('Two fixed prompt shards required')
        for i,lane in enumerate(group):
            if lane['slice']!=[16*i,16] or lane['gpu'] in seen or cpus.intersection(lane['cpus']):
                raise ValueError('Disjoint GPU/core assignments and complete fixed slices required')
            seen.add(lane['gpu']);cpus.update(lane['cpus'])
    return [(i,r) for i,r in enumerate(plan['requests']) if r['model'] in groups and r['batch']==1]


def aggregate(request, records, group, *, request_count=32):
    if len(records)!=len(group):raise ValueError('Every registered shard required')
    ordered=sorted(records,key=lambda r:r['report']['request_slice'][0]);reports=[v['report'] for v in ordered]
    cursor=0
    for record,lane in zip(ordered,group):
        value=record['report'];start,count=value['request_slice']
        if start!=cursor or count<1 or len(value['requests'])!=count:
            raise ValueError('Overlapping, missing or shortened prompt slices')
        check_result(value,request,lane['gpu'],lane['cpus'])
        if len(value['samples'])!=count*3:raise ValueError('Three repeats per original prompt required')
        cursor+=count
    if cursor!=request_count:raise ValueError('Full original prompt count required')
    fields=('job_sha256','mode','batch_size','decode_steps','cache_bytes','complete_serving_bytes','base_model_bytes',
            'source_row_bytes','decoded_row_sha256','phase_source_sha256','codec_adapter_source_sha256','cached_decode_source_sha256')
    for value in reports[1:]:
        if any(value.get(k)!=reports[0].get(k) for k in fields):raise ValueError('Shard model, bytes or runtime changed')
    result=copy.deepcopy(reports[0]);result.pop('gpu_before');result.pop('gpu_samples')
    result.update(measurement_mode='matched-prompt-shards',shards=ordered,request_slice=[0,request_count],
        selected_gpu_selector='matched-prompt-shards',cpu_affinity=[],
        requests=[v for r in reports for v in r['requests']],samples=[v for r in reports for v in r['samples']],
        unique_prompt_count=len({p for r in reports for v in r['requests'] for p in v['prompt_hashes']}),
        open_seconds=max(r['open_seconds'] for r in reports),
        aggregation_contract='Identical fixed prompt-to-GPU assignment for every method; per-request latency, not summed multi-GPU throughput')
    prefill=[v['prefill_ms'] for v in result['samples']];decode=[d for v in result['samples'] for d in v['decode_ms']]
    def percentile(values,p):
        values=sorted(values);at=(len(values)-1)*p;lo=int(at);hi=min(lo+1,len(values)-1)
        return values[lo]+(at-lo)*(values[hi]-values[lo])
    result.update(prefill_p50_ms=statistics.median(prefill),prefill_p95_ms=percentile(prefill,.95),
        decode_p50_ms=statistics.median(decode),decode_p95_ms=percentile(decode,.95),
        decode_tokens_per_second=request['batch']*1000/statistics.mean(decode))
    return result


def route_pending_shard(args):
    plan_path=args.output.parent.parent/'gpu-extra-shard-plan.json'
    if not plan_path.exists():return False
    plan=load(plan_path)
    if str(args.output.parent)!=plan['canonical_directory']:return False
    request=plan['requests'][int(args.output.stem)]
    if request['model'] not in plan['groups'] or request['batch']!=1:return False
    if (str(args.job),args.mode,args.cache_bytes,args.batch_size,args.requests,args.repeat,args.warmup,args.decode_steps)!=(
            request['job'],request['mode'],request['cache'],request['batch'],32,3,1,16):
        raise ValueError('Frozen original request or budget changed')
    if request['mode']=='independent' and str(args.control_run)!=request['control_run']:raise ValueError('Independent control changed')
    from .tree_node_quality import write
    target=plan_path.parent/'gpu-extra-shards/outputs'/args.output.name;deadline=time.monotonic()+24*3600
    while not target.exists():
        controller=plan_path.parent/'gpu-extra-shards/report.json'
        if controller.exists() and load(controller)['status']=='failed':raise RuntimeError('Prompt shard controller failed; diagnostics retained')
        if time.monotonic()>deadline:raise TimeoutError('Prompt-shard wait expired; workers retained')
        time.sleep(5)
    value=load(target)
    if value['status']!='passed' or value['job_sha256']!=digest(args.job):raise ValueError('Complete matched shards required')
    cpus=sorted(os.sched_getaffinity(0))
    if cpus!=plan['original_cpu_affinity'][request['model']]:raise ValueError('Original route core assignment changed')
    receipt={'status':'passed','measurement_purpose':'extra-shard-routing-receipt-only','job_sha256':digest(args.job),
        'cpu_affinity':cpus,'request':request,'extra_shard_plan_sha256':digest(plan_path),
        'shard_measurement_output':str(target),'shard_measurement_sha256':digest(target)}
    # Older drain controllers validate request configuration with check_result.
    # These are dispatcher fields, with no latency/memory/NVML measurement.
    # Publication replaces the entire cohort with the actual child reports.
    receipt.update(mode=request['mode'],batch_size=1,decode_steps=16,cache_bytes=request['cache'],
        selected_gpu_selector=str(plan['original_devices'][request['model']]),
        gpu_before={'compute_pids':[], 'scope':'Routing process has no CUDA client; not a physical GPU snapshot'},
        gpu_samples=[],routing_configuration_only=True)
    if args.output.exists():
        if load(args.output)!=receipt:raise ValueError('Completed original evidence must never be overwritten')
    else:write(args.output,receipt)
    return True


def run(plan_path):
    from .tree_node_quality import environment,write
    plan=load(plan_path);expected=targets(plan)
    if socket.gethostname()!=plan['measurement_host']:raise ValueError('Wrong physical measurement host')
    root=plan_path.parent;folder=root/'gpu-extra-shards';(folder/'outputs').mkdir(parents=True,exist_ok=True)
    report={'status':'running','environment':environment(),'measurement_host':socket.gethostname(),
        'requests':[r for _,r in expected],'complete':[],'preflights':[],'failures':[],
        'plan_sha256':digest(plan_path),'phase_source_sha256':digest(__file__),'workers':4,'workers_per_gpu':1}
    lock=threading.Lock();path=folder/'report.json';write(path,report)
    def measure(request,output,lane,preflight=False):
        cmd=command(request,output,plan['upstream'],repeat=1 if preflight else 3,warmup=0 if preflight else 1)
        cmd+=['--request-offset',str(lane['slice'][0]),'--request-count',str(2 if preflight else lane['slice'][1])]
        cmd=['taskset','--cpu-list',','.join(map(str,lane['cpus'])),*cmd]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane['gpu']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        output.parent.mkdir(parents=True,exist_ok=True)
        wait_idle(str(lane['gpu']),time.monotonic()+24*3600)
        result=run_gpu_job(cmd,env=env,output=output,log=output.with_suffix('.'+str(time.time_ns())+'.log'))
        if result.returncode:raise RuntimeError('Prompt shard worker failed: '+str(output))
        value=load(output);check_result(value,request,lane['gpu'],lane['cpus'])
        if value['measurement_host']!=plan['measurement_host']:raise ValueError('A shard ran on a different host')
        return {'report':value,'output':str(output),'sha256':digest(output)}
    def lane_work(item):
        model,lane=item;gpu=lane['gpu']
        source=next(r for r in plan['requests'] if r['model']==model and r['name']=='source-gpu' and r['batch']==8)
        native=next(r for r in plan['requests'] if r['model']==model and r['name']=='native-none-fp32' and r['batch']==8 and r['cache']==0)
        values=[measure(r,folder/'preflights'/f'{model}-gpu{gpu}-{name}.json',lane,True) for r,name in ((source,'source'),(native,'native'))]
        for a,b in zip(values[0]['report']['requests'],values[1]['report']['requests']):
            if any(a[k]!=b[k] for k in ('input_ids_sha256','cached_logit_hashes','generated_ids_sha256')):raise ValueError('New card source/native preflight changed bytes')
        with lock:report['preflights'].append({'model':model,'gpu':gpu,'records':values});write(path,report)
        for index,request in expected:
            if request['model']!=model:continue
            measure(request,folder/'shards'/f'{index:05d}-gpu{gpu}.json',lane)
            with lock:
                outputs=[folder/'shards'/f'{index:05d}-gpu{other["gpu"]}.json' for other in plan['groups'][model]]
                final=folder/'outputs'/f'{index:05d}.json'
                if all(p.exists() and load(p).get('status')=='passed' for p in outputs) and not final.exists():
                    records=[{'report':load(p),'output':str(p),'sha256':digest(p)} for p in outputs]
                    value=aggregate(request,records,plan['groups'][model]);value['extra_shard_plan_sha256']=digest(plan_path)
                    value['aggregation_source_sha256']=digest(__file__);write(final,value)
                if final.exists() and not any(v['request']==request for v in report['complete']):
                    report['complete'].append({'request':request,'output':str(final),'sha256':digest(final)})
                    write(path,report);print({'complete':len(report['complete']),'expected':len(expected)},flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lane_work,[(m,lane) for m,group in plan['groups'].items() for lane in group]))
        if len(report['complete'])!=len(expected):raise ValueError('Full prompt-sharded cohort required')
        report.update(status='complete',finished=time.time());write(path,report)
        write(root/plan['gpu_finished_marker'],{'status':'complete','measurement_host':socket.gethostname(),
            'finished':report['finished'],'controller_sha256':digest(path)})
    except BaseException as error:report['status']='failed';report['failures'].append({'error':repr(error)});raise
    finally:write(path,report)


def publish_cohort(root,timer,values,archive):
    plan_path=root/'gpu-extra-shard-plan.json'
    if not plan_path.exists():return timer,values
    from .publish_memory_application import check_controller
    plan=load(plan_path);expected=targets(plan)
    if plan['requests']!=timer['requests']:raise ValueError('Prompt sharding changed original schedule')
    controller,results=check_controller(root/'gpu-extra-shards',[r for _,r in expected])
    if controller['plan_sha256']!=digest(plan_path):raise ValueError('Prompt shard plan changed')
    archive.take(plan_path,'gpu-extra-shard-plan.json')
    archive.take(Path('benchmarks/memory_application_idle_gpu_addendum_v1.json'),'idle-gpu-addendum.json')
    archive.take(root/'gpu-extra-shards/report.json','llm/extra-shard-controller.json')
    archive.put('llm/pre-extra-sharded-gpu-controller.json',timer)
    archive.put('llm/pre-extra-sharded-gpu-records.json.gz',[{'request':v['request'],'private_sha256':v['sha256'],
        'report':values[json.dumps(v['request'],sort_keys=True)]} for v in timer['complete'] if v['request']['model'] in plan['groups'] and v['request']['batch']==1])
    replacement={json.dumps(v['request'],sort_keys=True):v for v in controller['complete']}
    complete=[replacement.get(json.dumps(v['request'],sort_keys=True),v) for v in timer['complete']]
    final={**timer,'complete':complete,'workers':16,'extra_shard_plan_sha256':digest(plan_path),
        'extra_shard_groups':plan['groups'],'extra_shard_host':plan['measurement_host'],
        'cohort_contract':'Complete model cohorts with identical per-prompt hardware assignments for every method'}
    archive.put('llm/retiming-gpu-controller.json',final)
    return final,{**values,**results}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    run(parser.parse_args().plan)


if __name__=='__main__':main()
