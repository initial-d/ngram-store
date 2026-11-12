#!/bin/bash
set -eu
TASK=${RUN_ROOT}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEFT_SOURCE_COMMIT=04b644f002e4dbcfd0791084f9cb1a65505fb688
PY=${ENV_ROOT}/bin/python
cd "$TASK/row-audit-source"
while ! test -f "$TASK/frontier-audit-complete"; do sleep 10; done
"$PY" -m tools.audit_tree_node_rows --frontier "$TASK/frontier" --output "$TASK/row-audit" --jobs 24 > "$TASK/row-audit.log" 2>&1
touch "$TASK/row-audit-complete"
