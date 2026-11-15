"""Reuse the registered classifier tables and official test IDs; CPU only."""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--commit', required=True)
args=parser.parse_args(); work=args.work
root=Path('/llm-align/duyimin/ngram-controlled-deployment-20261111')
bundles=Path('/llm-align/duyimin/ngram-host-profile-20261112/bundles')
(work/'traces').mkdir(exist_ok=True)
env=dict(os.environ, PYTHONPATH=str(work/'runtime'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
traces={}
for dataset in ('ag_news','dbpedia_14'):
 data=root/'data'/dataset
 ids=np.load(data/'test-ids.npy',mmap_mode='r'); offsets=np.load(data/'test-offsets.npy',mmap_mode='r')
 pick=np.sort(np.random.default_rng(20261112).choice(len(offsets)-1,size=256,replace=False))
 queries={}; records=[]
 for i,doc in enumerate(pick):
  q=np.ascontiguousarray(ids[offsets[doc]:offsets[doc+1]],dtype=np.uint64); q=q[q!=0]
  queries['ids'+str(i)]=q
  records.append({'document_index':int(doc),'ids_sha256':hashlib.sha256(q.tobytes()).hexdigest()})
 path=work/'traces'/f'{dataset}.npz';np.savez(path,**queries)
 metadata={'domain':f'Official {dataset} test documents, original registered row IDs; OOV zero removed as in StagedClassifier. 256 seeded documents (20261112).','trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'requests':records,'dataset_source':str(data)}
 path.with_suffix('.json').write_text(json.dumps(metadata,indent=2));traces[dataset]=path
jobs=[]
for dataset in ('ag_news','dbpedia_14'):
 for init in ('zero','uniform'):
  lr=10 if dataset=='ag_news' or init=='zero' else 3
  model=f'{dataset}-exact-{init}-s3-lr{lr}'
  table=bundles/model/'dedup-raw/rows.meft'
  cmd=[sys.executable,'-m','tools.benchmark_native_gather','--table',str(table),'--codes',str(root/'training-registered'/model/'rows.npy'),'--traces',str(traces[dataset]),'--baseline-library',str(work/'lib/baseline-codecs.so'),'--library',str(work/'lib/candidate-codecs.so'),'--output',str(work/'results'/f'{model}-raw.json'),'--commit',args.commit,'--core',str(40+len(jobs)*2),'--cache-bytes','0']
  jobs.append((model,cmd))
def run(job):
 model,cmd=job
 with (work/'results'/f'{model}-raw.log').open('w') as out:
  p=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,cwd=work/'runtime')
 print(model,p.returncode,flush=True)
 return {'model':model,'returncode':p.returncode,'command':cmd}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(run,jobs))
(work/'results/run-raw-status.json').write_text(json.dumps(results,indent=2))
