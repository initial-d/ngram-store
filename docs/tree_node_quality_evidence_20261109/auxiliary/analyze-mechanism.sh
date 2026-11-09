#!/bin/bash
set -eu
TASK=${RUN_ROOT}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEFT_SOURCE_COMMIT=0a23eee254551772ffc6057ea971daf8e70b7b42
PY=${ENV_ROOT}/bin/python
cd "$TASK/analysis-source"
while ! test -f "$TASK/mechanism-complete"; do sleep 10; done
"$PY" - <<'PY'
from pathlib import Path
from tools.tree_node_quality import load,write
p=Path('${RUN_ROOT}')
t=p/'mechanism/order-12-seed-3/legacy-init-1'
r=load(t/'report.json'); point=next(x for x in r['checkpoints'] if x['steps']==8000)
job={'checkpoint':str(t/'step-8000/reference.pt'),'point_record':point,'convention':r['convention'],'scale':r['ngram_initialization_scale'],'training_root':'${TRAINING_ROOT}','common':str(p/'frontier/legacy-order-12-seed-3/common'),'native_library':str(p/'frontier-source/build/libmeft_rows.so'),'structural_library':str(p/'frontier-source/build/libmeft_structural.so'),'output':str(p/'mechanism-preflight')}
write(p/'mechanism-preflight-job.json',job)
PY
"$PY" -m tools.analyze_tree_node_mechanism --worker "$TASK/mechanism-preflight-job.json" > "$TASK/mechanism-preflight.log" 2>&1
touch "$TASK/mechanism-preflight-complete"
"$PY" -m tools.analyze_tree_node_mechanism --mechanism "$TASK/mechanism" --frontier "$TASK/frontier" --training-root ${TRAINING_ROOT} --native-library "$TASK/frontier-source/build/libmeft_rows.so" --structural-library "$TASK/frontier-source/build/libmeft_structural.so" --output "$TASK/mechanism-analysis" --jobs 24 > "$TASK/mechanism-analysis.log" 2>&1
touch "$TASK/mechanism-analysis-complete"
