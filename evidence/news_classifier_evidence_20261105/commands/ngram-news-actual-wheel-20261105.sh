#!/bin/bash
set -euo pipefail
TASK_ROOT=/llm-align/duyimin/ngram-news-app-20261105
PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
cd "$TASK_ROOT/rss-source"
"$PYTHON" -m pip wheel . --no-index --no-deps --no-build-isolation --wheel-dir "$TASK_ROOT/app-wheel" > "$TASK_ROOT/actual-wheel-build.log" 2>&1
"$PYTHON" -m pip install --no-index --no-deps --target "$TASK_ROOT/installed-news" "$TASK_ROOT"/app-wheel/*.whl > "$TASK_ROOT/actual-wheel-install.log" 2>&1
mkdir "$TASK_ROOT/outside-checkout"
cd "$TASK_ROOT/outside-checkout"
unset MEFT_ROWS_LIBRARY
export PYTHONPATH="$TASK_ROOT/installed-news"
"$PYTHON" - <<'PY' > "$TASK_ROOT/actual-wheel-inference.log" 2>&1
from pathlib import Path
import hashlib,json,subprocess,sys
import ngram_store
from ngram_store import TextClassifier
root = Path('/llm-align/duyimin/ngram-news-app-20261105')
assert Path(ngram_store.__file__).is_relative_to(root/'installed-news')
assert 'torch' not in sys.modules
texts = ['The team won the final championship match.','Scientists announced a new computer chip.']
reports = {}
for backend in ('ngram-int8','ngram-bounded'):
    store = root/'serving-rss/relu/seed-3'/backend
    with TextClassifier(store) as model:
        result = model.predict(texts,require_preserved=backend=='ngram-bounded')
        assert 'torch' not in sys.modules
        reports[backend] = {'labels':[model.classes[int(i)] for i in result['predictions']],
            'probabilities':result['probabilities'].tolist(),
            'source_prediction_preserved':result['source_prediction_preserved'].tolist(),
            'classifier_sha256':hashlib.sha256((store/'classifier.json').read_bytes()).hexdigest()}
    cli = [sys.executable,'-m','ngram_store','classify','--store',str(store),'--text',*texts]
    if backend=='ngram-bounded': cli.append('--require-preserved')
    completed = subprocess.run(cli,capture_output=True,text=True,check=True)
    value = json.loads(completed.stdout)
    assert [v['label'] for v in value['predictions']]==reports[backend]['labels']
report = {'status':'passed','source_commit':'237cb86953a4a0061e1047b5a82d4c8d2d3d00cc',
    'package_path':ngram_store.__file__,'torch_loaded':'torch' in sys.modules,
    'scope':'Actual AG News-trained seed-3 bundles, installed wheel outside checkout; authored-text functional check, not new accuracy evidence.',
    'wheel_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'app-wheel').glob('*.whl')},
    'results':reports}
(root/'public-metadata/actual-installed-inference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
tail -n 12 "$TASK_ROOT/actual-wheel-inference.log"
