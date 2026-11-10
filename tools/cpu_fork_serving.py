"""Fresh CPU processes forked from a single-threaded, dependency-only parent.

No reader, model, query, CUDA context or result is reused. Unix IPC avoids
repeated NumPy/FAISS imports outside the measured query interval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def socket_path(plan,phase):
    host=hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]
    return Path(plan).parent/('cf-'+phase+'-'+host+'.sock')


def encode_args(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


def validate(request,plan_path,phase):
    args=SimpleNamespace(**request['args'])
    for name in ('job','output','plan','upstream'):
        if getattr(args,name,None) is not None:setattr(args,name,Path(getattr(args,name)))
    if args.phase!=phase or args.plan.resolve()!=plan_path.resolve():raise ValueError('Fork worker changed phase or plan')
    plan=json.loads(plan_path.read_text())
    if plan.get('measurement_host')!=socket.gethostname():raise ValueError('Wrong CPU measurement host')
    if args.output.parent.resolve() not in ((plan_path.parent/phase).resolve(),(plan_path.parent/phase/'serial-reference').resolve()):
        raise ValueError('Fork output outside registered timing directories')
    if args.output.exists() or args.lane not in plan['lanes']:raise ValueError('Fresh output and registered lane required')
    json.loads(args.job.read_text());family='long-frontier' if phase=='retime-long' else 'llm-frontier'
    if not args.job.resolve().is_relative_to((plan_path.parent/family).resolve()):raise ValueError('Job outside original frontier')
    model=args.job.resolve().relative_to((plan_path.parent/family).resolve()).parts[0]
    if plan['model_lanes'][phase][model]!=args.lane:raise ValueError('Original model/core assignment changed')
    if args.cache not in (0,1048576) or args.requests!=32:raise ValueError('Original CPU request/cache budgets required')
    return args,plan


def submit(args):
    path=socket_path(args.plan,args.phase)
    if not path.exists():return False
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(1800);client.connect(str(path))
        client.sendall((json.dumps({'args':encode_args(args),'argv':sys.argv})+'\n').encode())
        response=b''
        while b'\n' not in response:
            part=client.recv(65536)
            if not part:raise RuntimeError('Fork worker exited without an IPC result')
            response+=part
    result=json.loads(response.split(b'\n',1)[0])
    if result['status']!='complete':raise RuntimeError('Fresh fork worker failed: '+result.get('error','')+'; '+result.get('log',''))
    if not args.output.exists():raise ValueError('Fork worker omitted its output')
    return True


def serve(plan_path,phase,deadline_hours=18):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only dependency parent required')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Single-thread dependency parent required')
    import faiss
    faiss.omp_set_num_threads(1)
    from . import tree_node_quality,benchmark_faiss_cotraining,structural_native
    if phase=='retime-llm-cpu':
        import torch
        torch.set_num_threads(1)
        from . import benchmark_llm_codec_serving
    if Path('/proc/self/task').exists() and len(list(Path('/proc/self/task').iterdir()))!=1:
        raise ValueError('Fork parent must have no native worker threads')
    from .parallel_cpu_serving import worker
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    os.environ['MEFT_SOURCE_COMMIT']=commit
    worker_hash=digest(Path(__file__).with_name('parallel_cpu_serving.py'))
    service_hash=digest(__file__);plan_hash=digest(plan_path);path=socket_path(plan_path,phase)
    if path.exists():raise ValueError('An existing fork socket must be diagnosed before replacement')
    deadline=time.monotonic()+deadline_hours*3600;children=set()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(path));os.chmod(path,0o600);server.listen(64)
        print(json.dumps({'status':'ready','socket':str(path),'commit':commit,'phase':phase}),flush=True)
        try:
            while time.monotonic()<deadline:
                for pid in list(children):
                    if os.waitpid(pid,os.WNOHANG)[0]:children.remove(pid)
                controller=plan_path.parent/phase/'report.json'
                if not children and controller.exists() and json.loads(controller.read_text())['status']=='complete':break
                if not select.select([server],[],[],.2)[0]:continue
                client,_=server.accept();client.settimeout(10);payload=b''
                while b'\n' not in payload:
                    part=client.recv(65536)
                    if not part:raise RuntimeError('Incomplete fork request')
                    payload+=part
                    if len(payload)>65536:raise ValueError('Oversized fork request')
                request=json.loads(payload.split(b'\n',1)[0])
                try:
                    if digest(plan_path)!=plan_hash or digest(Path(__file__).with_name('parallel_cpu_serving.py'))!=worker_hash:
                        raise ValueError('Immutable fork source or CPU plan changed')
                    args,plan=validate(request,plan_path,phase)
                except Exception as error:
                    client.sendall((json.dumps({'status':'failed','error':repr(error)})+'\n').encode());client.close();continue
                sys.stdout.flush();sys.stderr.flush();pid=os.fork()
                if pid:
                    children.add(pid);client.close();continue
                server.close();log=args.output.with_suffix('.fork-'+str(os.getpid())+'.log')
                try:
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('w') as stream:
                        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
                        os.sched_setaffinity(0,set(plan['lanes'][args.lane]['cpus']));sys.argv=request['argv']
                        worker(args)
                        value=json.loads(args.output.read_text())
                        value['process_launch']={'kind':'fresh-fork','parent_commit':commit,
                            'source_sha256':service_hash,'fork_pid':os.getpid(),
                            'contract':'Dependency-only parent; a fresh process, table reader and query state for every measurement'}
                        tree_node_quality.write(args.output,value)
                    result={'status':'complete','output_sha256':digest(args.output),'log':str(log)}
                except BaseException as error:
                    traceback.print_exc();result={'status':'failed','error':repr(error),'log':str(log)}
                try:client.sendall((json.dumps(result)+'\n').encode())
                finally:client.close();os._exit(0 if result['status']=='complete' else 1)
        finally:
            # No evaluation child is stopped when the parent finishes.
            for pid in children:os.waitpid(pid,0)
            path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--phase',choices=('retime-long','retime-llm-cpu'),required=True)
    args=parser.parse_args();serve(args.plan,args.phase)


if __name__=='__main__':main()
