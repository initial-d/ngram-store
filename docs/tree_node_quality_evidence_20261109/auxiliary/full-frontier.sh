#!/bin/bash
set -eu
TASK=${RUN_ROOT}
PY=${ENV_ROOT}/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MEFT_SOURCE_COMMIT=b58c3fa31ce73cb2dff36963a98f4b2a9e9503b4
cd "$TASK/frontier-source"
test -f "$TASK/preflight-complete"
"$PY" -m tools.tree_node_quality build-grid --output "$TASK/frontier" --jobs 24 > "$TASK/frontier-grid.log" 2>&1
printf '%s\n' complete > "$TASK/frontier-build-complete"
"$PY" -m tools.tree_node_quality audit-grid --output "$TASK/frontier" --jobs 24 > "$TASK/frontier-audit.log" 2>&1
printf '%s\n' complete > "$TASK/frontier-audit-complete"
