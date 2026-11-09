#!/bin/bash
set -eu
TASK=${RUN_ROOT}
PY=${ENV_ROOT}/bin/python
trap 'kill -CONT 2179768' EXIT
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MEFT_SOURCE_COMMIT=76901984cc55dfed36715228529fec89b1270179
export MEFT_QUERY_STRUCTURAL_LIBRARY="$TASK/query-repair-source/build/libmeft_structural.so"
cd "$TASK/query-repair-source"
"$PY" -m tools.audit_tree_node_inputs --frontier "$TASK/frontier" --training-root ${TRAINING_ROOT} --corpus-zip ${CORPUS_ZIP} --output "$TASK/input-audit" > "$TASK/input-audit.log" 2>&1
touch "$TASK/input-integrity-precheck-complete"
