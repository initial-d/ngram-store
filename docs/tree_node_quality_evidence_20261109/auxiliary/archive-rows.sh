#!/bin/bash
set -eu
TASK=${RUN_ROOT}
while ! test -f "$TASK/publication-complete" || ! test -f "$TASK/row-audit-complete"; do sleep 10; done
${ENV_ROOT}/bin/python - <<'PY'
import gzip,hashlib,json
from pathlib import Path
p=Path('${RUN_ROOT}')
out=p/'public-evidence'
roots=[(str(p),'${RUN_ROOT}'),('${ENV_ROOT}','${ENV_ROOT}')]
def public(v):
 if isinstance(v,dict):return {k:public(x) for k,x in v.items()}
 if isinstance(v,list):return [public(x) for x in v]
 if isinstance(v,str):
  for a,b in roots:v=v.replace(a,b)
  if '/llm-align/' in v:raise ValueError('unredacted path')
 return v
controller=p/'row-audit/report.json'
r=json.loads(controller.read_text());assert r['status']=='complete' and len(r['complete'])==3672
r['private_controller_sha256']=hashlib.sha256(controller.read_bytes()).hexdigest()
(out/'row-audit-controller.json').write_text(json.dumps(public(r),indent=2,sort_keys=True)+'\n')
models={}
for item in r['complete']:
 path=p/'row-audit'/item['model']/(item['name']+'.json')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['report_sha256']
 models.setdefault(item['model'],[]).append({'private_report_sha256':item['report_sha256'],'report':public(json.loads(path.read_text()))})
for model,reports in models.items():
 target=out/'models'/model/'row-fidelity.json.gz'
 target.write_bytes(gzip.compress((json.dumps(sorted(reports,key=lambda x:x['report']['name']),sort_keys=True,allow_nan=False)+'\n').encode(),mtime=0))
for name in ('full-frontier-runner.log','frontier-grid.log','frontier-audit.log','mechanism.log','mechanism-analysis.log','row-audit.log','retiming.log','publication.log','finalize-v3.sh','row-audit-v2.sh'):
 path=p/name
 if path.exists():
  (p/'public-auxiliary'/name).write_text(public(path.read_text()))
PY
touch "$TASK/rows-publication-complete"
