"""Full fixed-core reference cohort under explicitly recorded concurrent load.

The legacy 'serial-reference' path/cohort identifies the reference request set.
Execution is eight concurrent lanes, each serial on its original fixed core.
This does not claim host-exclusive CPU latency.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from .parallel_cpu_serving import digest,load,reference_requests,check_sample


def wait_for_reference(args):
    plan=args.plan.parent/'cpu-parallel-reference-plan.json'
    if getattr(args,'cohort',None)!='serial-reference' or not plan.exists():return False
    value=load(plan)
    if value['measurement_host']!=socket.gethostname():raise ValueError('Wrong parallel reference host')
    controller=plan.parent/'cpu-parallel-references'/args.phase/'report.json';deadline=time.monotonic()+24*3600
    while not args.output.exists():
        if controller.exists() and load(controller)['status']=='failed':raise RuntimeError('Parallel reference failed; diagnostics retained')
        if time.monotonic()>deadline:raise TimeoutError('Parallel reference wait expired')
        time.sleep(.2)
    report=load(args.output)
    if report['reference_execution']['plan_sha256']!=digest(plan):raise ValueError('Reference execution plan changed')
    return True


def worker(args):
    from .parallel_cpu_serving import worker as original_worker
    from .tree_node_quality import write
    plan=args.plan.parent/'cpu-parallel-reference-plan.json';registered=load(plan)
    if args.output.exists() or args.cohort!='serial-reference' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh CPU-only reference output required')
    if registered['measurement_host']!=socket.gethostname() or digest(args.plan)!=registered['original_cpu_plan_sha256']:
        raise ValueError('Original reference hardware plan changed')
    started=time.time();original_worker(args);value=load(args.output)
    value['reference_execution']={'kind':'parallel-fixed-core','workers':registered['workers'],
        'plan_sha256':digest(plan),'source_sha256':digest(__file__),'started':started,
        'contract':'One fresh single-thread worker per fixed core; concurrent reference lanes share host memory bandwidth'}
    write(args.output,value)


def run(root,phase):
    from .tree_node_quality import environment,write
    path=root/'cpu-parallel-reference-plan.json';plan=load(path)
    if socket.gethostname()!=plan['measurement_host']:raise ValueError('Wrong reference host')
    original_plan=root/'cpu-serving-parallel-plan.json';hardware=load(original_plan)
    if digest(original_plan)!=plan['original_cpu_plan_sha256']:raise ValueError('Original CPU plan changed')
    folder=root/'cpu-parallel-references'/phase;folder.mkdir(parents=True,exist_ok=True)
    report={'status':'waiting-for-complete-main-grid-and-local-gpu','phase':phase,'environment':environment(),
        'workers':plan['workers'],'plan_sha256':digest(path),'phase_source_sha256':digest(__file__),
        'complete':[],'failures':[],'requests':plan['phases'][phase]['requests']}
    out=folder/'report.json';write(out,report);deadline=time.monotonic()+24*3600
    primary=root/phase/'report.json';marker=root/hardware['local_gpu_finished_marker']
    while True:
        if primary.exists():
            timer=load(primary)
            if timer['status']=='failed':raise RuntimeError('Main CPU grid failed; references remain blocked')
            if len(timer['complete'])==len(timer['requests']) and marker.exists():
                gate=load(marker)
                if gate['status']!='complete' or gate['measurement_host']!=socket.gethostname():raise ValueError('Wrong reference GPU gate')
                break
        if time.monotonic()>deadline:raise TimeoutError('Reference main/GPU gate wait expired')
        time.sleep(5)
    family='long-frontier' if phase=='retime-long' else 'llm-frontier'
    selection=load(root/family/'selection.json')
    if reference_requests(timer['requests'],selection,phase)!=report['requests']:raise ValueError('Original full reference set changed')
    report.update(status='running',started=time.time(),gpu_gate_sha256=digest(marker));write(out,report)
    lock=threading.Lock();failed=threading.Event()
    def lane_work(lane):
        for i,request in enumerate(report['requests']):
            if failed.is_set():return
            if hardware['model_lanes'][phase][request['model']]!=lane:continue
            target=root/phase/'serial-reference'/f'{i:05d}.json';target.parent.mkdir(exist_ok=True)
            args={'phase':phase,'job':request['job'],'output':str(target),'plan':str(original_plan),
                'lane':lane,'cohort':'serial-reference','cache':request['cache'],'requests':32,
                'upstream':plan['upstream']}
            if phase=='retime-long':args.update(cached=request['cached'],index=request['index'])
            cmd=['taskset','--cpu-list',','.join(map(str,hardware['lanes'][lane]['cpus'])),sys.executable,
                 '-m','tools.cpu_parallel_references','worker']
            for k,v in args.items():cmd+=['--'+k,str(v)]
            try:
                if not target.exists():
                    log=folder/(f'{i:05d}.'+str(time.time_ns())+'.log')
                    with log.open('w') as stream:
                        result=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError('Reference worker failed: '+str(log))
                value=load(target);check_sample(value,request,plan['original_cpu_plan_sha256'],lane,hardware['lanes'][lane]['cpus'],'serial-reference')
                if value['reference_execution']['plan_sha256']!=digest(path):raise ValueError('Reference was not executed under registered concurrent load')
                with lock:
                    report['complete'].append({'request':request,'output':str(target),'sha256':digest(target)})
                    if len(report['complete'])%20==0:write(out,report);print(phase,len(report['complete']),len(report['requests']),flush=True)
            except BaseException as error:
                failed.set()
                with lock:report['status']='failed';report['failures'].append({'request':request,'error':repr(error)});write(out,report)
                raise
    try:
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:list(pool.map(lane_work,sorted(hardware['lanes'])))
        if len(report['complete'])!=len(report['requests']):raise ValueError('Complete original reference cohort required')
        report.update(status='complete',finished=time.time())
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:write(out,report)


def qualify(root,timer,phase,archive):
    path=root/'cpu-parallel-reference-plan.json'
    if not path.exists():return timer
    plan=load(path);controller=root/'cpu-parallel-references'/phase/'report.json';value=load(controller)
    if value['status']!='complete' or value['requests']!=timer['reference_requests'] or value['plan_sha256']!=digest(path):
        raise ValueError('Complete unchanged concurrent references required')
    expected={json.dumps(r,sort_keys=True) for r in timer['reference_requests']}
    actual={json.dumps(r['request'],sort_keys=True) for r in value['complete']}
    if expected!=actual or len(actual)!=len(value['complete']):raise ValueError('Reference coverage changed')
    hashes={json.dumps(v['request'],sort_keys=True):v['sha256'] for v in timer['serial_references']}
    for item in value['complete']:
        if digest(item['output'])!=item['sha256'] or hashes[json.dumps(item['request'],sort_keys=True)]!=item['sha256']:
            raise ValueError('Concurrent reference output changed')
    archive.take(controller,'diagnostics/'+phase+'-parallel-reference-controller.json')
    archive.take(root/phase/'report.json','diagnostics/'+phase+'-legacy-serial-dispatch-controller.json')
    return {**timer,'serial_reference_workers':plan['workers'],'reference_execution':'parallel-fixed-core',
        'parallel_reference_plan_sha256':digest(path),'reference_claim':'Concurrent fixed-core references; host memory bandwidth is shared'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','worker'))
    parser.add_argument('--phase',choices=('retime-long','retime-llm-cpu'),required=True)
    parser.add_argument('--root',type=Path)
    for name in ('job','output','plan','upstream'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--lane');parser.add_argument('--cohort',default='serial-reference')
    parser.add_argument('--cache',type=int,default=0);parser.add_argument('--requests',type=int,default=32)
    parser.add_argument('--cached',type=int,default=0);parser.add_argument('--index',default='datrie')
    args=parser.parse_args()
    if args.mode=='run':
        try:run(args.root,args.phase)
        except BaseException as error:
            path=args.root/'cpu-parallel-references'/args.phase/'report.json'
            if path.exists():
                report=load(path);report.update(status='failed',error=repr(error));path.write_text(json.dumps(report,indent=2)+'\n')
            raise
    else:worker(args)


if __name__=='__main__':main()
