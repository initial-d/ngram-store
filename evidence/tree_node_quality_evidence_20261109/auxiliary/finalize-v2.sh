#!/bin/bash
set -eu
TASK=${RUN_ROOT}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEFT_SOURCE_COMMIT=0fd0f447915f111952c42de392830584123e83c5
PY=${ENV_ROOT}/bin/python
cd "$TASK/publication-source"
while ! test -f "$TASK/frontier-audit-complete" || ! test -f "$TASK/mechanism-analysis-complete" || ! test -f "$TASK/mechanism-complete"; do sleep 10; done
"$PY" -m tools.finalize_tree_node_quality retime --frontier "$TASK/frontier" --output "$TASK/retiming" > "$TASK/retiming.log" 2>&1
touch "$TASK/retiming-complete"
"$PY" -m tools.finalize_tree_node_quality publish --frontier "$TASK/frontier" --mechanism "$TASK/mechanism" --analysis "$TASK/mechanism-analysis" --retiming "$TASK/retiming" --output "$TASK/public-evidence" > "$TASK/publication.log" 2>&1
touch "$TASK/publication-complete"
