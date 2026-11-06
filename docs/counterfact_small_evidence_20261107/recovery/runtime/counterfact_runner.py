"""Finite six-GPU experiment queue. Never signals other users' processes.

The separate, task-scoped occupancy helper must be explicitly authorized and
audited before invocation. This runner only starts its own child jobs, records
their outcomes, and stops that helper when its finite queue ends.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .counterfact_capacity import write,sha
from .counterfact_resume import completed_task

GPUS=['GPU-805ec4aa-37cf-6d18-f341-da44657c3844','GPU-7c1fe9b0-4b49-e530-ebab-5cb72ec71aa4',
      'GPU-739011e2-6861-76a1-aacc-ec382e95e733','GPU-20d75cd9-0182-957b-65df-9a2d44291c6e',
      'GPU-0e8947d0-b90d-8e5a-06dd-342ef5db3cc8','GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8']
MODELS={'qwen06':'/llm-align/open_models/Qwen3/Qwen3-0.6B','qwen35':'/llm-align/open_models/Qwen3.5/Qwen3.5-4B'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Hash-validate and reuse completed artifacts; preserve previous outcomes')
    args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to('/llm-align/duyimin'):raise ValueError('user directory only')
    protocol=json.loads(args.protocol.read_text());status={'status':'waiting-for-preflight','protocol_sha256':sha(args.protocol),'jobs':[]}
    if (root/'queue.json').exists():
        if not args.resume:raise ValueError('existing queue requires explicit --resume')
        previous=json.loads((root/'queue.json').read_text())
        if previous['status'] not in ('complete','failed') or previous['protocol_sha256']!=status['protocol_sha256']:
            raise ValueError('resume requires a stopped queue with the same protocol')
        if any('exit_code' not in job and 'pid' in job and Path('/proc',str(job['pid'])).exists() for job in previous['jobs']):
            raise ValueError('a previous queue child is still active; wait before resuming')
        history=root/'queue-history';history.mkdir(exist_ok=True)
        snapshot=history/(str(time.time_ns())+'.json');snapshot.write_bytes((root/'queue.json').read_bytes())
        status.update(previous_queue=str(snapshot),previous_queue_sha256=sha(snapshot))
    write(root/'queue.json',status);(root/'runs').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    active={};failed=False
    def jobs_for(edits):
        tasks=[]
        for model in MODELS:
            tasks.append({'action':'baseline','model':model,'edits':edits,'name':f'{model}-{edits}-unedited-base'})
        # Pair controls across architectures before moving to the next seed.
        for seed in protocol['seeds']:
            for variant in ('off','large','small','mid','adapter','head-constant'):
                for model in MODELS:
                    tasks.append({'action':'train','model':model,'edits':edits,'seed':seed,'variant':variant,'name':f'{model}-{edits}-{variant}-s{seed}'})
        for model in MODELS:tasks.append({'action':'retrieval','model':model,'edits':edits,'name':f'{model}-{edits}-bm25'})
        return tasks
    def execute(tasks):
        nonlocal failed
        pending=[]
        for task in tasks:
            reused=completed_task(root,task,status['protocol_sha256']) if args.resume else None
            if reused:status['jobs'].append(reused)
            else:pending.append(task)
        write(root/'queue.json',status)
        while pending or active:
            if (root/'queue-stop').exists():raise RuntimeError('user-requested queue stop marker')
            for gpu in list(active):
                p,job,out=active[gpu]
                if p.poll() is None:continue
                out.close();job['exit_code']=p.returncode;job['finished_at']=time.time();del active[gpu]
                if p.returncode:failed=True
            if failed and not active:raise RuntimeError('job failure; remaining dependent work was not started')
            for gpu in GPUS:
                if not pending or failed:break
                if gpu in active:continue
                task=pending.pop(0);model=task['model'];edits=task['edits'];action=task['action']
                command=[sys.executable,'-m','tools.counterfact_capacity',action,'--base',MODELS[model]]
                command+=['--output',str(root/'runs'/task['name'])] if action!='cache' else ['--output',str(root/f'cache-{model}-{edits}')]
                if action=='retrieval':command+=['--data',str(root/'data/multi_counterfact.json'),'--edits',str(edits),'--protocol',str(args.protocol)]
                else:
                    command+=['--upstream','/llm-align/duyimin/meft-cuda-20261030/upstream','--prepared',str(root/f'prepared-{model}-{edits}')]
                    if action=='cache':command+=['--shard',str(task['shard']),'--shards','6','--suffix',str(protocol['suffix_layers'])]
                    else:command+=['--cache',str(root/f'cache-{model}-{edits}'),'--protocol',str(args.protocol)]
                    if action=='train':command+=['--seed',str(task['seed']),'--variant',task['variant']]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu}
                log=root/'logs'/(task['name']+'.log');out=log.open('w')
                p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                job={**task,'pid':p.pid,'gpu':gpu,'command':command,'log':str(log),'started_at':time.time()}
                status['jobs'].append(job);active[gpu]=(p,job,out)
            write(root/'queue.json',status);time.sleep(5)
    try:
        # Existing preflight/cache jobs are outside this process. Wait for their
        # explicit completion manifests before using any of the same GPUs.
        deadline=time.monotonic()+7200
        while True:
            ready=True
            for model in MODELS:
                pre=root/f'{model}-preflight.json'
                if not pre.exists():ready=False;continue
                report=json.loads(pre.read_text())
                if report['status']!='passed':raise RuntimeError('preflight failed: '+model)
                shards=2 if model=='qwen06' else 3
                if len(list((root/f'cache-{model}-1024').glob('shard-*.json')))!=shards:ready=False
            if ready:break
            if time.monotonic()>deadline:raise TimeoutError('preflight/cache deadline')
            time.sleep(5)
        status['status']='small-edit-grid';write(root/'queue.json',status);execute(jobs_for(1024))
        for model in MODELS:
            if not (root/f'prepared-{model}-20877/prepare.json').exists():raise RuntimeError('full dataset preparation incomplete')
        status['status']='full-edit-prefix-cache';write(root/'queue.json',status)
        execute([{'action':'cache','model':m,'edits':20877,'shard':s,'name':f'{m}-20877-cache{s}'} for m in MODELS for s in range(6)])
        status['status']='full-edit-grid';write(root/'queue.json',status);execute(jobs_for(20877))
        status['status']='complete'
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        # No helper should survive after this finite experiment ends. Existing
        # own children are allowed to finish; unknown processes are untouched.
        (root/'compute-finished').touch();write(root/'queue.json',status)


if __name__=='__main__':main()
