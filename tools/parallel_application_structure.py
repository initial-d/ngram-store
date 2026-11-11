"""Complete remaining structural gates concurrently using the frozen planner.

Existing builders finish their current gate before administrative replacement.
No training or evaluation process is stopped. Already published jobs stay intact.
"""
import argparse
from collections import OrderedDict
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from .large_application_frontier import evaluate, publish_job
from .native_row_store import NativeRows, pack_native
from .structural_native import StructuralPlan
from .tree_node_quality import digest, load, memory, serving_files, write

GATES = [('exact', 1.), *[('joint', t) for t in (.99, .98, .97, .95, .90)]]


def gate(args):
    model=args.model; source=Path(load(model/'jobs/direct-fp32.json')['source'])
    reference=load(model/'jobs/direct-fp32.json');data=Path(reference['data'])
    name=f'datrie-{args.mode}-{args.cosine:g}'
    stage=model/'parallel-structure'/name;stage.mkdir(parents=True,exist_ok=True)
    directory=stage/'bundle';directory.mkdir(exist_ok=True)
    rows=np.load(source/'rows.npy',mmap_mode='r')
    registry=np.load(data/'registry.npy',mmap_mode='r');order=load(data/'prepared.json')['ngram_order']
    tokens=registry.view('<u4').reshape(-1,order)
    keys={i+1:''.join(f'{int(v):08x}|' for v in token if v) for i,token in enumerate(tokens)}
    started=time.perf_counter();before=memory()
    with StructuralPlan.from_array(rows,keys,args.structural_library,mode=args.mode,cosine=args.cosine,norm_ratio=.99) as plan:
        if plan.source_sha256 != load(source/'training.json')['rows_sha256']:
            # training digest includes the NPY header; direct-fp32 decoded hash is raw rows.
            if plan.source_sha256 != reference['decoded_sha256']:raise ValueError('source rows changed')
        artifact=directory/'rows.meft'
        if not artifact.exists():
            pack_native(model/'source.raw',artifact,rows.shape[1]*4,256,False,'xor',
                representatives=plan.representatives,row_order=plan.row_order)
        import hashlib
        sha=hashlib.sha256()
        with NativeRows(artifact,args.library) as reader:
            for begin in range(0,len(rows),8192):
                end=min(begin+8192,len(rows));actual=reader.rows(range(begin,end))
                if actual!=rows[plan.representatives[begin:end]].tobytes():raise ValueError('representative bytes changed')
                sha.update(actual)
        if sha.hexdigest()!=plan.reconstructed_sha256:raise ValueError('reconstructed digest changed')
        structural={'mode':args.mode,'cosine':args.cosine,'norm_ratio':.99,
            'retained_rows':sum(i==r for i,r in enumerate(plan.representatives)),
            'original_bytes_preserved':plan.original_bytes_preserved,'source_sha256':plan.source_sha256,
            'reconstructed_sha256':plan.reconstructed_sha256,'planner_library_sha256':digest(args.structural_library),
            'borrowed_readonly_rows':True,'construction_seconds':time.perf_counter()-started,
            'memory_before':before,'memory_after':memory(),
            'key_contract':'Training-only fixed-width token fields; retained proper prefix/suffix containment checked by the original DATrie callbacks and final direct cosine/norm guard.'}
        write(directory/'structural.json',structural)
        reconstructed=plan.reconstructed_sha256
    del keys
    final=model/'bundles'/name
    if final.exists():
        if any(final.iterdir()):raise ValueError('original builder reached reserved downstream gate')
        final.rmdir()
    directory.replace(final)
    files=serving_files(final);shared=reference['common_files']
    job={'name':name,'directory':str(final),'kind':'native','source':str(source),'data':str(data),
        'source_report_sha256':digest(source/'training.json'),'rows':len(rows),'dim':rows.shape[1],
        'decoded_sha256':reconstructed,'library':str(args.library),'library_sha256':digest(args.library),
        'common_files':shared,'complete_model_bytes':sum(v['bytes'] for v in shared.values())+sum(v['bytes'] for v in files.values()),
        'files':files,'quality_reference':name,'source_sha256':digest(__file__)}
    staged_job=stage/(name+'.json');publish_job(staged_job,job)
    # Quality precedes job publication so the main controller cannot duplicate it.
    for split in ('calibration','test'):
        evaluate(argparse.Namespace(job=staged_job,split=split,output=model/'quality'/(name+'-'+split+'.json')))
    staged_job.replace(model/'jobs'/(name+'.json'))
    write(stage/'complete.json',{'status':'complete','job_sha256':digest(model/'jobs'/(name+'.json')),'source_sha256':digest(__file__)})


def run(args):
    root=args.root;app=root/'application-frontier';output=app/'parallel-structure';output.mkdir(exist_ok=True)
    main=load(app/'dispatch.json');builders=[];running={};done={};failed={}
    for name,item in main['running'].items():
        if not name.startswith('build-dbpedia_14-'):continue
        command=item['command'];model=Path(command[command.index('--output')+1])
        missing=[(mode,threshold) for mode,threshold in GATES if not (model/'jobs'/f'datrie-{mode}-{threshold:g}.json').exists()]
        if not missing:continue
        current=missing[0];builders.append({'pid':item['pid'],'model':str(model),'current':current,'stopped':False})
        for mode,threshold in missing[1:]:
            name=model.name+f'-{mode}-{threshold:g}';core=26+len(running)
            command=['taskset','-c',str(core),sys.executable,'-m','tools.parallel_application_structure','gate',
                '--model',str(model),'--mode',mode,'--cosine',str(threshold),'--library',str(args.library),
                '--structural-library',str(args.structural_library)]
            with (output/(name+'.log')).open('a') as log:
                child=subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
                    'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            running[name]={'child':child,'pid':child.pid,'command':command,'model':str(model)}
    while True:
        for builder in builders:
            if builder['stopped']:continue
            mode,threshold=builder['current'];model=Path(builder['model'])
            if (model/'jobs'/f'datrie-{mode}-{threshold:g}.json').exists():
                proc=Path('/proc')/str(builder['pid'])
                if proc.exists():
                    command=(proc/'cmdline').read_bytes()
                    if b'tools.large_application_frontier\x00build' not in command or str(model).encode() not in command:
                        raise RuntimeError('construction PID identity changed')
                    os.kill(builder['pid'],signal.SIGTERM)
                builder['stopped']=True;builder['stopped_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        for name,item in list(running.items()):
            code=item['child'].poll()
            if code is None:continue
            value={k:v for k,v in item.items() if k!='child'};value['exit_code']=code
            (done if code==0 else failed)[name]=value;del running[name]
        for builder in builders:
            model=Path(builder['model']);jobs=sorted((model/'jobs').glob('*.json'))
            if builder['stopped'] and len(jobs)==88 and not (model/'built.json').exists():
                write(model/'built.json',{'status':'complete','jobs':list(map(str,jobs)),
                    'selection_sha256':digest(app/'source-selection.json'),'fit_ids_sha256':digest(model/'fit-ids.npy'),
                    'source_sha256':digest(__file__),'parallel_structure_manifest':str(output/'dispatch.json')})
        finished=not running and all((Path(b['model'])/'built.json').exists() for b in builders)
        write(output/'dispatch.json',{'status':('failed' if failed else 'complete') if finished else 'running',
            'controller_pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'builders':builders,'running':{n:{k:v for k,v in x.items() if k!='child'} for n,x in running.items()},
            'completed':done,'failed':failed,'source_sha256':digest(__file__)})
        if finished:return
        if failed:raise RuntimeError('a parallel gate failed; outputs and original current gates retained')
        time.sleep(2)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    c=sub.add_parser('run');c.add_argument('--root',type=Path,required=True)
    c2=sub.add_parser('gate');c2.add_argument('--model',type=Path,required=True)
    c2.add_argument('--mode',choices=('exact','joint'),required=True);c2.add_argument('--cosine',type=float,required=True)
    for parser in (c,c2):
        parser.add_argument('--library',type=Path,required=True);parser.add_argument('--structural-library',type=Path,required=True)
    args=p.parse_args();(run if args.phase=='run' else gate)(args)

if __name__=='__main__':main()
