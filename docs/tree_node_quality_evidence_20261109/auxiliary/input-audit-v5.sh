#!/bin/bash
set -eu
TASK=${RUN_ROOT}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEFT_SOURCE_COMMIT=4a94b974df3f06f2baef705bece68450f285ea35
export MEFT_QUERY_STRUCTURAL_LIBRARY="$TASK/query-repair-source/build/libmeft_structural.so"
PY=${ENV_ROOT}/bin/python
cd "$TASK/input-repair-source"
while ! test -f "$TASK/retiming-complete"; do sleep 10; done
if ! test -f "$TASK/input-integrity-precheck-complete"; then
"$PY" -m tools.audit_tree_node_inputs --frontier "$TASK/frontier" --training-root ${TRAINING_ROOT} --corpus-zip ${CORPUS_ZIP} --output "$TASK/input-audit" > "$TASK/input-audit.log" 2>&1
fi
while ! test -f "$TASK/publication-complete"; do sleep 10; done
"$PY" - <<'PY'
from pathlib import Path
from tools.tree_node_quality import load,write,digest
p=Path('${RUN_ROOT}')
a=load(p/'input-audit/report.json');assert a['status']=='complete'
a['private_report_sha256']=digest(p/'input-audit/report.json')
def redact(v):
 if isinstance(v,dict):return {k:redact(x) for k,x in v.items()}
 if isinstance(v,list):return [redact(x) for x in v]
 if isinstance(v,str):
  for old,new in ((str(p),'${RUN_ROOT}'),('${TRAINING_ROOT}','${TRAINING_ROOT}'),('${ENV_ROOT}','${ENV_ROOT}'),('${CORPUS_ZIP}','${CORPUS_ZIP}')):v=v.replace(old,new)
  assert '/llm-align/' not in v
 return v
write(p/'public-evidence/input-integrity.json',redact(a))
PY
touch "$TASK/input-audit-complete"
