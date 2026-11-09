#!/bin/bash
set -eu
TASK=${RUN_ROOT}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEFT_SOURCE_COMMIT=ef065dc9da1b9bfb583613b8bcd3ab9c59c58d22
PY=${ENV_ROOT}/bin/python
cd "$TASK/row-repair-v2-source"
"$PY" - <<'PY'
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,sys,shutil
from tools.tree_node_quality import load,write,digest,environment
p=Path('${RUN_ROOT}');out=p/'row-audit'
assert not (out/'report.initial-failed.json').exists()
shutil.copyfile(out/'report.json',out/'report.initial-failed.json')
r=load(out/'report.json');assert len(r['failures'])==108 and len(r['complete'])==3564
failures=r['failures'];r['initial_failure_report_sha256']=digest(out/'report.initial-failed.json');r['repair_environment']=environment();r['repaired']=[];r['repair_failures']=[]
def execute(f):
 job=load(f['job']);model=Path(job['common']).parent.name;name=job['name'];output=out/model/(name+'.json')
 assert not output.exists()
 with (out/model/(name+'.retry.log')).open('w') as log:
  process=subprocess.run([sys.executable,'-m','tools.audit_tree_node_rows','--job',f['job'],'--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
 assert process.returncode==0,(model,name,process.returncode)
 return {'model':model,'name':name,'report_sha256':digest(output)}
with ThreadPoolExecutor(max_workers=24) as pool:
 for future in as_completed([pool.submit(execute,f) for f in failures]):
  try:r['repaired'].append(future.result())
  except Exception as e:r['repair_failures'].append({'error':repr(e)})
  write(out/'repair-progress.json',r)
  print({'repaired':len(r['repaired']),'failures':len(r['repair_failures'])},flush=True)
assert len(r['repaired'])==108 and not r['repair_failures']
r['complete'].extend(r['repaired']);r['failures']=[];r['status']='complete';write(out/'report.json',r)
PY
touch "$TASK/row-audit-complete"
