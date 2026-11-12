set -eu
ROOT=/llm-align/duyimin/ngram-counterfact-capacity-20261106
cd "$ROOT/source-native"
export PYTHONPATH="$ROOT/source-native" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$ROOT/env/bin/python" - <<'PY'
from pathlib import Path
import json,time
r=Path('/llm-align/duyimin/ngram-counterfact-capacity-20261106');deadline=time.monotonic()+172800
while True:
 q=json.loads((r/'queue.json').read_text())
 if q['status'] in ('complete','failed'):break
 if time.monotonic()>deadline:raise TimeoutError('grid postprocessing deadline')
 time.sleep(10)
PY
"$ROOT/env/bin/python" -m tools.counterfact_analyze --runs "$ROOT/runs" --output "$ROOT/final-analysis.json"
"$ROOT/env/bin/python" - <<'PY'
from pathlib import Path
import gzip,json,hashlib,shutil
r=Path('/llm-align/duyimin/ngram-counterfact-capacity-20261106');out=r/'evidence-archive';out.mkdir(exist_ok=True)
manifest={}
for name in ('queue.json','final-analysis.json','hardware.json','precision-audit.json','model-manifest.json'):
 shutil.copyfile(r/name,out/name)
for p in sorted((r/'runs').glob('*/*')):
 if not p.name.startswith(('case-scores','choice-scores','report')) or not p.name.endswith('.json'):continue
 target=out/(p.parent.name+'-'+p.name+'.gz')
 with p.open('rb') as source,target.open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as dest:shutil.copyfileobj(source,dest)
 manifest[target.name]={'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                       'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(r/'postprocessing-finished').write_text('complete\n')
PY
