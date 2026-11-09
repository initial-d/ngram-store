"""Frozen quality choices and serial fresh-process serving comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from .tree_node_quality import BUDGETS, digest, environment, load, query_job, select, write


def long_requests(root):
    prepared=load(root/'prepare.json'); frozen=load(root/'selection.json')
    complete=load(root/'quality-complete.json')
    if (complete['status']!='complete' or complete['models']!=108 or complete['requested_candidates']!=28188
            or complete['frozen_selection_sha256']!=digest(root/'selection.json')):
        raise ValueError('complete frozen long-training quality required')
    requested=[]
    for model,selected in zip(prepared['models'],frozen['models']):
        if model['name']!=selected['name'] or len(model['jobs'])!=261:raise ValueError('long grid changed')
        jobs={load(p)['name']:(p,load(p)) for p in model['jobs']}
        builds={name:load(job['report']) for name,(_,job) in jobs.items()}
        for name,built in builds.items():
            if digest(jobs[name][1]['report'])!=selected['costs'][name]['raw_build_report_sha256']:
                raise ValueError('frozen build changed')
        for choice in selected['choices']:
            candidates=[{**r,'model_bytes':selected['costs'][r['name']]['complete_serving_bytes']}
                for r in builds.values() if r['family']==choice['family']]
            best=select(candidates,choice['budget'],choice['cached'])
            if choice['name']!=(best['name'] if best else None):raise ValueError('frozen minimum-byte choice changed')
        pairs={(c['name'],c['cached']) for c in selected['choices'] if c['name']}
        for name in ('raw-fp32','raw-fp16','raw-int8','dense-fp32','native-none-fp32','native-none-fp16','native-none-int8',
                     'constant-zero','constant-mean','constant-weighted-mean'):
            pairs.update((name,cached) for cached in (0,1))
        for name,cached in sorted(pairs):
            path,job=jobs[name]
            for index in ('datrie','hash'):
                for cache in ((0,1048576) if job['backend']=='native' or job.get('native_codes_source') else (0,)):
                    requested.append({'model':model['name'],'name':name,'job':path,'cached':cached,'index':index,'cache':cache})
    random.Random(42).shuffle(requested)
    return requested


def llm_requests(root,controls):
    prepared=load(root/'prepare.json'); frozen=load(root/'selection.json'); requested=[]
    for model,selected in zip(prepared['models'],frozen['models']):
        folder=root/model['name']; ledger=load(folder/'test-ledger.json')
        if ledger['status']!='complete' or ledger['frozen_selection_sha256']!=digest(root/'selection.json'):
            raise ValueError('complete frozen LLM test audit required')
        jobs={load(p)['name']:(p,load(p)) for p in model['jobs']}
        names={c['name'] for c in selected['choices'] if c['name']}
        names.update(('raw-fp32','raw-fp16','raw-int8','dense-fp32','native-none-fp32','native-none-fp16','native-none-int8'))
        for name in sorted(names):
            path,job=jobs[name]
            for cache in ((0,1048576) if job['backend']=='native' or job.get('native_codes_source') else (0,)):
                for batch in (1,8):
                    requested.append({'model':model['name'],'name':name,'job':path,'mode':'codec','cache':cache,'batch':batch})
        for batch in (1,8):
            requested.append({'model':model['name'],'name':'source-gpu','job':jobs['raw-fp32'][0],
                'mode':'source','cache':0,'batch':batch})
            for control in ('head-constant','adapter'):
                base,seed=model['name'].split('-exact-s')
                run=controls/f'{base}-{control}-s{seed}'
                if load(run/'report.json')['status']!='complete':raise ValueError('complete independent control required')
                requested.append({'model':model['name'],'name':'trained-'+control,'job':jobs['raw-fp32'][0],
                    'mode':'independent','control_run':str(run),'cache':0,'batch':batch})
    random.Random(42).shuffle(requested)
    return requested


def serial(args):
    # The caller may finish GPU training while CPU codecs still run. Final
    # timers start only after the whole registered quality phase has completed.
    ready=load(args.ready)
    if ready['status']!='quality-complete':raise ValueError('all-quality completion marker required')
    llm=args.phase in ('retime-llm-cpu','retime-llm-gpu')
    requests=llm_requests(args.frontier,args.controls) if llm else long_requests(args.frontier)
    if args.phase=='retime-llm-gpu' and args.gpu_addendum:
        return gpu_parallel(args, requests, ready)
    if args.phase=='retime-llm-cpu':
        # CPU row/addressing has no independent hidden-state adapter queries.
        # Record each frozen codec/cache once, independent of GPU batch choice.
        requests=[r for r in requests if r['mode']=='codec' and r['batch']==1]
    args.output.mkdir(parents=True,exist_ok=True); controller=args.output/'report.json'
    selection_hash=digest(args.frontier/'selection.json')
    previous=load(controller) if controller.exists() else {}
    if previous and (previous['requests']!=requests or previous['frozen_selection_sha256']!=selection_hash):
        raise ValueError('existing serial schedule differs')
    report={'status':'running','environment':environment(),'phase_source_sha256':digest(__file__),
        'frozen_selection_sha256':selection_hash,'all_quality_ready_sha256':digest(args.ready),
        'phase':args.phase,'workers':1,'requests':requests,'complete':[],
        'failures':previous.get('failures',[]),'execution_schedule_sha256':hashlib.sha256(json.dumps(requests,sort_keys=True).encode()).hexdigest()}
    write(controller,report); current_failures=[]
    retained={Path(item['output']).name:item['sha256'] for item in previous.get('complete',[])}
    for index,request in enumerate(requests):
        output=args.output/(f'{index:05d}.json')
        if output.exists():
            if output.name in retained and digest(output)!=retained[output.name]:
                raise ValueError('previously completed serial evidence changed')
            value=load(output)
            if value.get('status') in ('passed','complete'):
                if value.get('job_sha256')!=digest(request['job']):raise ValueError('serial evidence uses a different job')
                report['complete'].append({'request':request,'output':str(output),'sha256':digest(output)});write(controller,report);continue
            output.rename(str(output)+'.failed-'+str(time.time_ns()))
        log=args.output/(f'{index:05d}.'+str(time.time_ns())+'.log')
        if llm:
            phase='gpu' if args.phase=='retime-llm-gpu' else 'cpu'
            command=[sys.executable,'-m','tools.benchmark_llm_codec_serving',phase,'--job',request['job'],
                '--output',str(output),'--upstream',str(args.upstream),'--cache-bytes',str(request['cache']),
                '--requests',str(args.requests),'--repeat','3','--warmup','1']
            if phase=='gpu':
                command+=['--mode',request['mode'],'--batch-size',str(request['batch']),'--decode-steps','16']
                if request['mode']=='independent':command+=['--control-run',request['control_run']]
        else:
            command=[sys.executable,'-m','tools.finalize_memory_application','query-long','--job',request['job'],
                '--output',str(output),'--cached',str(request['cached']),'--index',request['index'],'--cache',str(request['cache'])]
        with log.open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            failed={'request':request,'returncode':result.returncode,'log':str(log),'time':time.time()}
            current_failures.append(failed);report['failures'].append(failed)
        else:
            value=load(output)
            if value['status'] not in ('passed','complete'):raise ValueError('timer did not write complete evidence')
            report['complete'].append({'request':request,'output':str(output),'sha256':digest(output)})
        write(controller,report)
        print({'phase':args.phase,'complete':len(report['complete']),'expected':len(requests),'current_failures':len(current_failures)},flush=True)
    if digest(args.frontier/'selection.json')!=selection_hash:raise ValueError('final serving changed selection')
    report['status']='complete' if not current_failures and len(report['complete'])==len(requests) else 'failed'
    write(controller,report)
    if report['status']!='complete':raise RuntimeError('serial measurement grid incomplete')


def gpu_parallel(args, requests, ready):
    """One fresh worker per device, all methods of a seed on that device."""
    amendment=load(args.gpu_addendum)
    assignments=amendment['device_assignment']
    models=sorted({r['model'] for r in requests})
    if set(models)!=set(assignments) or len(set(assignments.values()))!=len(models):
        raise ValueError('Fixed disjoint model/GPU assignment required')
    allowed=sorted(os.sched_getaffinity(0)); chunks={}
    # Explicit CPU sets come from the recorded device/NUMA topology.
    affinity=load(args.gpu_affinity)
    used=set()
    for model in models:
        cpus=affinity[model]
        if not cpus or not set(cpus)<=set(allowed) or used.intersection(cpus):
            raise ValueError('Disjoint allowed GPU worker CPU affinity required')
        used.update(cpus);chunks[model]=cpus
    args.output.mkdir(parents=True,exist_ok=True);controller=args.output/'report.json'
    selection_hash=digest(args.frontier/'selection.json');lock=threading.Lock()
    previous=load(controller) if controller.exists() else {}
    if previous and (previous['requests']!=requests or previous['frozen_selection_sha256']!=selection_hash
                     or previous['gpu_addendum_sha256']!=digest(args.gpu_addendum) or previous['cpu_affinity']!=affinity):
        raise ValueError('Existing per-device schedule differs')
    report={'status':'running','environment':environment(),'phase':args.phase,'phase_source_sha256':digest(__file__),
            'frozen_selection_sha256':selection_hash,'all_quality_ready_sha256':digest(args.ready),
            'gpu_addendum_sha256':digest(args.gpu_addendum),'workers':len(models),'workers_per_gpu':1,
            'fixed_device_assignment':assignments,'cpu_affinity':affinity,'requests':requests,'complete':[],
            'failures':previous.get('failures',[]),'execution_schedule_sha256':hashlib.sha256(json.dumps(requests,sort_keys=True).encode()).hexdigest()}
    write(controller,report);retained={Path(e['output']).name:e['sha256'] for e in previous.get('complete',[])}
    current_failures=[]
    def lane(model):
        gpu=assignments[model]
        for index,request in enumerate(requests):
            if request['model']!=model:continue
            output=args.output/f'{index:05d}.json'
            if output.exists():
                if output.name in retained and digest(output)!=retained[output.name]:raise ValueError('Completed device evidence changed')
                result=load(output)
                if result.get('status')=='passed':
                    if result['job_sha256']!=digest(request['job']) or result['cpu_affinity']!=chunks[model]:raise ValueError('Retained worker differs')
                    with lock:
                        report['complete'].append({'request':request,'output':str(output),'sha256':digest(output),'gpu':gpu});write(controller,report)
                    continue
                output.rename(str(output)+'.failed-'+str(time.time_ns()))
            command=['taskset','--cpu-list',','.join(map(str,chunks[model])),sys.executable,'-m','tools.benchmark_llm_codec_serving',
                     'gpu','--job',request['job'],'--output',str(output),'--upstream',str(args.upstream),
                     '--mode',request['mode'],'--cache-bytes',str(request['cache']),'--batch-size',str(request['batch']),
                     '--requests',str(args.requests),'--decode-steps','16','--repeat','3','--warmup','1']
            if request['mode']=='independent':command+=['--control-run',request['control_run']]
            log=args.output/(f'{index:05d}.'+str(time.time_ns())+'.log')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with log.open('w') as stream:
                result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            with lock:
                if result.returncode:
                    failed={'request':request,'returncode':result.returncode,'log':str(log),'gpu':gpu,'time':time.time()}
                    report['failures'].append(failed);current_failures.append(failed)
                else:
                    value=load(output)
                    if value['status']!='passed' or value['cpu_affinity']!=chunks[model]:raise ValueError('GPU worker isolation differs')
                    report['complete'].append({'request':request,'output':str(output),'sha256':digest(output),'gpu':gpu})
                write(controller,report)
                print({'phase':args.phase,'complete':len(report['complete']),'expected':len(requests),'current_failures':len(current_failures),'gpu':gpu},flush=True)
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        list(pool.map(lane,models))
    if digest(args.frontier/'selection.json')!=selection_hash:raise ValueError('GPU serving changed frozen selection')
    report['status']='complete' if not current_failures and len(report['complete'])==len(requests) else 'failed'
    write(controller,report)
    if report['status']!='complete':raise RuntimeError('Per-device final serving grid incomplete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('retime-long','retime-llm-cpu','retime-llm-gpu','query-long'))
    for name in ('frontier','output','controls','ready','upstream','job'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--gpu-addendum',type=Path)
    parser.add_argument('--gpu-affinity',type=Path)
    parser.add_argument('--requests',type=int,default=32)
    parser.add_argument('--cached',type=int,choices=(0,1),default=0)
    parser.add_argument('--cache',type=int,default=0)
    parser.add_argument('--index',choices=('datrie','hash'),default='datrie')
    args=parser.parse_args()
    if args.phase!='retime-llm-gpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        parser.error('empty CUDA_VISIBLE_DEVICES required for CPU phases')
    if args.phase=='query-long':
        query_job(args.job,args.cached,index_kind=args.index,cache_bytes=args.cache,selected=True)
        job=load(args.job);result=Path(job['report']).with_name(job['name']+f'-cache{args.cached}-{args.index}-rowcache{args.cache}-retiming.json')
        write(args.output,load(result))
    else:serial(args)


if __name__=='__main__':main()
