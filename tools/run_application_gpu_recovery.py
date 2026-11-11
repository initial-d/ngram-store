"""Finish interrupted public-application GPU pairs without killing evaluations.

A recovery pair is one frozen source/cohort/batch/allocator-budget block with
every control and selected job. Only blocks already passed by the original
lane cursor are eligible while it runs. Failed attempts and old partial pairs
are archived; whole pairs run sequentially on a leased GPU and physical core.
"""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .run_large_application_frontier import acquire_gpu_lease, peer_reserved_gpus
from .tree_node_quality import digest, load, write


def registered_blocks(model, controls):
    frozen = load(model / 'selection.json'); built = load(model / 'built.json')
    by_name = {load(path)['name']: Path(path) for path in built['jobs']}
    names = {c['name'] for c in frozen['selection'].values() if c is not None}
    for name in list(names):
        reference = load(by_name[name])['quality_reference']
        if reference in by_name: names.add(reference)
    jobs = [*controls, *[by_name[name] for name in sorted(names)]]
    blocks = []; index = 0
    for cohort in range(3):
        rotated = jobs[cohort:] + jobs[:cohort]
        for batch in (1, 32):
            for budget in (.5, 1., 2., 4., 8.):
                cells = []
                for path in rotated:
                    job = load(path)
                    for mode in (('resident', 'host') if job['kind']=='source' else ('host',)):
                        name = f"c{cohort}-b{batch}-g{budget:g}-{job['name']}-{mode}"
                        cells.append({'name': name, 'job': str(path), 'job_sha256': digest(path),
                            'mode': mode, 'batch_size': batch, 'budget_gib': budget, 'ordinal': index,
                            'output': str(model / 'gpu' / (name + '.json'))}); index += 1
                blocks.append({'name': f'{model.name}-c{cohort}-b{batch}-g{budget:g}', 'model': str(model),
                    'cohort': cohort, 'batch_size': batch, 'budget_gib': budget,
                    'selection_sha256': digest(model / 'selection.json'), 'cells': cells})
    return blocks


def valid_block(block):
    try:
        return _valid_block(block)
    except (FileNotFoundError, __import__('json').JSONDecodeError):
        return False


def _valid_block(block):
    identities = set()
    for cell in block['cells']:
        path = Path(cell['output'])
        if not path.exists(): return False
        value = load(path)
        if value.get('status') not in ('passed', 'allocator-oom'): return False
        if value.get('job_sha256') != cell['job_sha256'] or digest(cell['job']) != cell['job_sha256']: return False
        if (value.get('mode'),value.get('batch_size'),value.get('allocator_budget_bytes')) != (
            cell['mode'],cell['batch_size'],int(cell['budget_gib']*(1<<30))): return False
        before = value.get('gpu_before', {})
        policy = value.get('resource_policy', 'exclusive')
        if policy == 'exclusive' and (before.get('external_compute_pids') or any(s.get('external_compute_pids') for s in value.get('gpu_samples',[]))): return False
        if policy not in ('shared', 'exclusive'): return False
        identities.add((before.get('gpu_uuid'),tuple(value.get('cpu_affinity',[])),value.get('source_sha256'),policy))
        if value['status']=='passed':
            payload=path.with_suffix('.npz')
            if not payload.exists() or digest(payload)!=value['payload_sha256']: return False
    return len(identities)==1 and all(next(iter(identities)))


def processed_cursor(progress, blocks):
    by_name={c['name']:c['ordinal'] for b in blocks for c in b['cells']}
    names=[Path(p).stem for p in progress.get('completed',[])]
    names.extend(Path(v['attempt']).stem.rsplit('-',1)[0] for v in progress.get('failed_attempts',[]))
    return max((by_name[n] for n in names if n in by_name),default=-1)


def run_block(args):
    block=load(args.block);model=Path(block['model'])
    if digest(model/'selection.json')!=block['selection_sha256']:raise ValueError('frozen selection changed')
    job=load(block['cells'][0]['job'])
    import numpy as np
    official=len(np.load(Path(job['data'])/'test-labels.npy',mmap_mode='r'))
    audited=set()
    for path in [*(model/'gpu').glob('*.json'), *(model/'gpu-quality-retained').glob('*.json')]:
        try:value=load(path)
        except (FileNotFoundError, __import__('json').JSONDecodeError):continue
        if value.get('status')=='passed' and value.get('documents')==official:
            if path.with_suffix('.npz').exists() and digest(path.with_suffix('.npz'))==value['payload_sha256']:
                audited.add((value['job_sha256'],value['mode']))
    output=args.block.parent/(args.block.stem+'-run');output.mkdir(parents=True,exist_ok=True)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(); staged=[]
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for cell in block['cells']:
        result=output/(cell['name']+'.json')
        full=(cell['batch_size']==32 and block['cohort']==0 and block['budget_gib']==8.
            and (cell['job_sha256'],cell['mode']) not in audited)
        command=[sys.executable,'-m','tools.benchmark_large_application','--job',cell['job'],'--output',str(result),
            '--mode',cell['mode'],'--budget-gib',str(cell['budget_gib']),'--batch-size',str(cell['batch_size'])]
        if not full:command.append('--timing-only')
        if args.allow_shared_gpu:command.append('--allow-shared-gpu')
        with result.with_suffix('.log').open('w') as log:
            code=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if code!=0 or not result.exists() or load(result)['status'] not in ('passed','allocator-oom'):
            raise RuntimeError('recovery cell failed; complete pair is retained unpublished')
        if full and load(result)['status']=='passed':audited.add((cell['job_sha256'],cell['mode']))
        staged.append({**cell,'output':str(result)})
    if not valid_block({**block,'cells':staged}):raise RuntimeError('recovered pair identity audit failed')
    releases=args.root/'inventory/occupancy-release-history.jsonl'
    if releases.exists():
        for line in releases.read_text().splitlines():
            event=__import__('json').loads(line)
            if event['utc']>=started:raise RuntimeError('known occupancy release overlapped recovery; replay whole pair')
    history=output/'original-pair';history.mkdir(parents=True,exist_ok=True)
    for cell,temporary in zip(block['cells'],staged):
        final=Path(cell['output']);source=Path(temporary['output']);final.parent.mkdir(parents=True,exist_ok=True)
        retain_quality=final.exists() and load(final).get('status')=='passed' and load(final).get('documents')==official
        for old in (final,final.with_suffix('.npz')):
            if old.exists():
                if retain_quality:
                    retained=model/'gpu-quality-retained';retained.mkdir(exist_ok=True)
                    shutil.copy2(old,retained/old.name)
                shutil.move(str(old),str(history/old.name))
        if source.with_suffix('.npz').exists():
            temporary=final.with_suffix('.npz.tmp');shutil.copy2(source.with_suffix('.npz'),temporary);temporary.replace(final.with_suffix('.npz'))
        temporary=final.with_suffix('.json.tmp');shutil.copy2(source,temporary);temporary.replace(final)
    write(output/'complete.json',{'status':'complete','started_utc':started,
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256':digest(__file__),'pair':block['name'],
        'reports':{c['output']:digest(c['output']) for c in block['cells']}})


def run(args):
    root=args.root;app=root/'application-frontier';output=app/'gpu-recovery';output.mkdir(parents=True,exist_ok=True)
    state_path=output/'dispatch.json';running={};completed={};failed=[];attempts={};all_blocks={}
    def live(item,model=None):
        try:command=Path(f"/proc/{item['pid']}/cmdline").read_bytes()
        except FileNotFoundError:return False
        return str(root).encode() in command and (model is None or str(model).encode() in command)
    if state_path.exists() and load(state_path).get('status')=='running':
        old=load(state_path)
        if live({'pid':old['controller_pid']}):raise RuntimeError('recovery controller already active')
    def save(status='running'):
        write(state_path,{'status':status,'controller_pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_sha256':digest(__file__),'running':{n:{k:v for k,v in x.items() if k not in ('child','log','lease')} for n,x in running.items()},
            'completed':completed,'failed_attempts':failed,'reserved_gpus':sorted(x['gpu'] for x in running.values()),
            'pairing':'One frozen source/cohort/batch/allocator-budget block; all methods sequentially use the same GPU and physical CPU core. No test reselection or cross-GPU absolute-latency comparison.'})
    save()
    while True:
        for name,item in list(running.items()):
            code=item['child'].poll()
            if code is None:continue
            item['log'].close();item['lease'].close()
            final={k:v for k,v in item.items() if k not in ('child','log','lease')};final['exit_code']=code
            if code==0:completed[name]=final
            else:failed.append(final)
            del running[name]
        main=load(app/'dispatch.json');selection=load(app/'source-selection.json');pending=[];models=list((app/'models').iterdir())
        for model in sorted(models):
            if not (model/'selection.json').exists():continue
            if model.name not in all_blocks:
                entry=next(e for e in selection['selected'] if Path(e['training']).name==model.name)
                controls=[app/'source-jobs'/(Path(e['training']).name+'.json') for e in selection['selected']
                    if e['family'][:2]==entry['family'][:2] and e['seed']==entry['seed']]
                all_blocks[model.name]=registered_blocks(model,controls)
            blocks=all_blocks[model.name];lane=main.get('running',{}).get('gpu-'+model.name)
            lane_live=lane is not None and live(lane,model)
            if lane_live:continue
            progress=load(model/'gpu-progress.json') if (model/'gpu-progress.json').exists() else {}
            cursor=10**9
            bad=[]
            for block in blocks:
                if block['name'] in running:bad.append(block);continue
                if valid_block(block):continue
                bad.append(block)
                if block['cells'][-1]['ordinal']<=cursor:pending.append(block)
            if not bad and not lane_live:
                marker=model/'gpu-complete.json'
                old=load(marker) if marker.exists() else {}
                if old.get('status')!='complete' or old.get('pairing_unit')!='source/cohort/batch/allocator-budget':
                    if old.get('status')!='complete':
                        archive=output/'original-completion'/model.name;archive.mkdir(parents=True,exist_ok=True)
                        saved=archive/'gpu-complete.json'
                        if marker.exists() and not saved.exists():shutil.copy2(marker,saved)
                    write(marker,{'status':'complete','expected':sum(len(b['cells']) for b in blocks),
                        'completed':[c['output'] for b in blocks for c in b['cells']],
                        'source_sha256':digest(__file__),'frozen_selection_sha256':digest(model/'selection.json'),
                        'pairing_unit':'source/cohort/batch/allocator-budget',
                        'retained_original_completion':str(output/'original-completion'/model.name/'gpu-complete.json'),
                        'recovery_manifest':str(state_path)})
        reserved={v['gpu'] for v in main.get('running',{}).values() if v.get('gpu') is not None and live(v)}
        reserved|={v['gpu'] for v in running.values()}
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        occupied={line.split(',')[0].strip() for line in raw.splitlines() if ',' in line}
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        for line in devices.splitlines():
            gpu,uuid,util=[v.strip() for v in line.split(',')];gpu=int(gpu)
            if not pending or gpu in reserved or (not args.allow_shared_gpu and (uuid in occupied or int(util)>5)):continue
            lease=acquire_gpu_lease(root,gpu)
            if lease is None:continue
            block=pending.pop(0);name=block['name'];attempts[name]=attempts.get(name,0)+1
            path=output/'blocks'/(name+f"-a{attempts[name]}-"+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')+'.json');write(path,block)
            logfile=path.with_suffix('.log');log=logfile.open('w')
            command=['taskset','-c',str(18+gpu),sys.executable,'-m','tools.run_application_gpu_recovery','block',
                '--root',str(root),'--block',str(path)]
            if args.allow_shared_gpu:command.append('--allow-shared-gpu')
            child=subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            running[name]={'child':child,'log':log,'lease':lease,'pid':child.pid,'gpu':gpu,'command':command,
                'block':str(path),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};save()
        done=len(models)==12 and all((m/'gpu-complete.json').exists() and load(m/'gpu-complete.json').get('status')=='complete' for m in models)
        if done and not running:
            save('complete')
            if main['status']=='failed' and not main['running']:
                archive=output/'original-terminal-dispatch.json';shutil.copy2(app/'dispatch.json',archive)
                command=[sys.executable,'-m','tools.run_large_application_frontier','run','--root',str(root),
                    '--library',str(args.library),'--structural-library',str(args.structural_library),
                    '--upstream',str(args.upstream),'--resume-dispatch',str(archive)]
                with (output/'terminal-controller.log').open('w') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                write(output/'terminal-controller.json',{'pid':child.pid,'command':command})
            return
        save();time.sleep(10)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    c=sub.add_parser('run')
    for name in ('root','library','structural-library','upstream'):c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--allow-shared-gpu',action='store_true')
    c=sub.add_parser('block');c.add_argument('--root',type=Path,required=True);c.add_argument('--block',type=Path,required=True)
    c.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();(run if args.phase=='run' else run_block)(args)


if __name__=='__main__':main()
