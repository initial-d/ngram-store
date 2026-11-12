"""Finite identity experiment; preserve old jobs and reject occupied GPU slots."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GPUS=['GPU-bc11cba9-e025-335f-83a8-9ec15f87a496','GPU-805ec4aa-37cf-6d18-f341-da44657c3844',
      'GPU-7c1fe9b0-4b49-e530-ebab-5cb72ec71aa4','GPU-739011e2-6861-76a1-aacc-ec382e95e733',
      'GPU-20d75cd9-0182-957b-65df-9a2d44291c6e','GPU-0e8947d0-b90d-8e5a-06dd-342ef5db3cc8',
      'GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8']
INDICES=dict(zip(GPUS,[4,1,2,3,5,6,7]))
MODELS={'qwen06':'/llm-align/open_models/Qwen3/Qwen3-0.6B','qwen35':'/llm-align/open_models/Qwen3.5/Qwen3.5-4B'}


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def occupied():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return {row[0].strip() for row in csv.reader(io.StringIO(output)) if len(row)>=2}


def old_reserved(root):
    q=json.loads((root/'queue.json').read_text())
    active={j['gpu'] for j in q['jobs'] if 'gpu' in j and 'exit_code' not in j and
            Path('/proc',str(j.get('pid',0))).exists()}
    # Until every original job is dispatched, its scheduler can use any old
    # slot at its next tick. Card 4 has never belonged to that original queue.
    if q['status'] not in ('complete','failed') and len(q['jobs'])<92:
        active.update(GPUS[1:])
    return active


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','old-root','protocol'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to('/llm-align/duyimin') or not args.old_root.resolve().is_relative_to('/llm-align/duyimin'):
        raise ValueError('user directories only')
    if (root/'queue.json').exists():raise ValueError('fresh queue required; preserve previous outcomes')
    protocol=json.loads(args.protocol.read_text());state={'status':'preflight','jobs':[]};active={};failed=False
    (root/'logs').mkdir(exist_ok=True);(root/'runs').mkdir(exist_ok=True);deadline=time.monotonic()+172800
    def execute(tasks):
        nonlocal failed
        pending=list(tasks)
        while pending or active:
            if time.monotonic()>deadline:raise TimeoutError('finite experiment deadline')
            for gpu in list(active):
                p,job,stream=active[gpu]
                if p.poll() is None:continue
                stream.close();job.update(exit_code=p.returncode,finished_at=time.time());del active[gpu]
                failed|=p.returncode!=0
            if failed and not active:raise RuntimeError('job failure; dependent work not started')
            reserved=old_reserved(args.old_root)
            candidates=[g for g in GPUS if g not in reserved and g not in active]
            scope=set(active)|set(candidates[:len(pending)] if not failed else [])
            write(root/'release-scope.json',[INDICES[g] for g in GPUS if g in scope])
            busy=occupied()
            for gpu in candidates:
                if not pending or failed:break
                if gpu in busy:continue  # Only the audited helper may clear occupiers.
                task=pending.pop(0);model=task['model'];edits=task.get('edits',1024)
                common=['--base',MODELS[model],'--upstream','/llm-align/duyimin/meft-cuda-20261030/upstream',
                        '--prepared',str(root/f'prepared-{model}-{edits}'),'--cache',str(root/f'cache-{model}-{edits}')]
                if task['action']=='check':
                    command=[sys.executable,'-m','tools.counterfact_identity','check',*common,
                             '--vocabulary',str(root/f'vocab-{model}-{edits}/vocabulary.npz'),'--output',str(root/task['name'])]
                else:
                    command=[sys.executable,'-m','tools.counterfact_capacity','train',*common,
                             '--identity-vocabulary',str(root/f'vocab-{model}-{edits}/vocabulary.npz'),
                             '--protocol',str(args.protocol),'--seed',str(task['seed']),'--variant',task['variant'],
                             '--output',str(root/'runs'/task['name'])]
                log=root/'logs'/(task['name']+'.log');stream=log.open('w')
                p=subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdin=subprocess.DEVNULL,
                                   stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                job={**task,'gpu':gpu,'pid':p.pid,'command':command,'log':str(log),'started_at':time.time()}
                state['jobs'].append(job);active[gpu]=(p,job,stream)
            write(root/'queue.json',state);time.sleep(3)
    try:
        execute([{'action':'check','model':m,'name':f'{m}-preflight'} for m in MODELS])
        for m in MODELS:
            if json.loads((root/f'{m}-preflight/report.json').read_text())['status']!='passed':raise RuntimeError('preflight failed')
        state['status']='training-grid';write(root/'queue.json',state)
        execute([{'action':'train','model':m,'edits':e,'seed':s,'variant':v,'name':f'{m}-{e}-{v}-s{s}'}
                 for e in protocol['edit_counts'] for s in protocol['seeds'] for m in MODELS for v in ('hash','exact')])
        state['status']='complete'
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        (root/'compute-finished').touch();write(root/'release-scope.json',[]);write(root/'queue.json',state)


if __name__=='__main__':main()
