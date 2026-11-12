set -eu
ROOT=/llm-align/duyimin/ngram-counterfact-capacity-20261106
cd "$ROOT/source-recovery"
export PYTHONPATH="$ROOT/source-recovery" HF_HOME="$ROOT/cache/hf" XDG_CACHE_HOME="$ROOT/cache"
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
"$ROOT/env/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import sys,json,shutil
r=Path(sys.argv[1]);v=json.loads((r/'recovery-validation.json').read_text())
assert v['status']=='validated' and len(v['verified_jobs'])==46
assert not (r/'queue-stop').exists()
for name in ['queue.pid','release-helper.pid','postprocess-grid.pid']:
    pid=(r/name).read_text().strip();assert not Path('/proc',pid).exists(),name
for name in ['compute-finished','postprocessing-finished']:
    (r/name).unlink(missing_ok=True)
(r/'release-scope.json').write_text('[1,2,3,5,6,7]\n')
text=(r/'postprocess-grid.sh').read_text().replace('source-native','source-recovery')
(r/'postprocess-resumed-grid.sh').write_text(text)
PY
nohup "$ROOT/env/bin/python" "$ROOT/release-occupiers.py" > "$ROOT/release-helper-resume.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/release-helper.pid"
nohup "$ROOT/env/bin/python" -m tools.counterfact_runner --resume --root "$ROOT" \
  --protocol "$ROOT/source-recovery/benchmarks/counterfact_capacity_v1.json" \
  > "$ROOT/queue-resume.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/queue.pid"
"$ROOT/env/bin/python" - "$ROOT" <<'PY'
from pathlib import Path
import sys,json,time
r=Path(sys.argv[1]);deadline=time.monotonic()+30
while time.monotonic()<deadline:
    q=json.loads((r/'queue.json').read_text())
    if q.get('previous_queue'):break
    time.sleep(1)
else:raise TimeoutError('resumed queue did not publish its new state')
print('resumed queue',q['status'],'pid',(r/'queue.pid').read_text().strip())
PY
nohup bash "$ROOT/postprocess-resumed-grid.sh" > "$ROOT/postprocess-resume.log" 2>&1 < /dev/null &
echo "$!" > "$ROOT/postprocess-grid.pid"
