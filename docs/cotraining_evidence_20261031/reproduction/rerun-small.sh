#!/usr/bin/env bash
set -euo pipefail
: "${PY:?Set PY to a Python with Torch and NumPy}"
: "${CORPUS:?Set CORPUS to the cached public text}"
: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the nine cached checkpoints}"
: "${NATIVE_LIBRARY:?Set NATIVE_LIBRARY to the built C row library}"
: "${STRUCTURAL_LIBRARY:?Set STRUCTURAL_LIBRARY to the built DATrie bridge}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new directory}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$OUTPUT_ROOT"
"$PY" -m tools.benchmark_cotraining --corpus "$CORPUS" \
  --checkpoint-root "$CHECKPOINT_ROOT" --dim 128 \
  --native-library "$NATIVE_LIBRARY" --structural-library "$STRUCTURAL_LIBRARY" \
  --output-dir "$OUTPUT_ROOT/cached-results"
for convention in legacy average; do
  "$PY" -m tools.benchmark_cotraining --corpus "$CORPUS" \
    --conventions "$convention" --native-library "$NATIVE_LIBRARY" \
    --structural-library "$STRUCTURAL_LIBRARY" \
    --output-dir "$OUTPUT_ROOT/dynamics-$convention"
done
"$PY" -m tools.benchmark_cotraining_layout --checkpoint-root "$CHECKPOINT_ROOT" \
  --native-library "$NATIVE_LIBRARY" --output-dir "$OUTPUT_ROOT/layout-small"
"$PY" -m tools.benchmark_cotraining_planner --checkpoint-root "$CHECKPOINT_ROOT" \
  --native-library "$NATIVE_LIBRARY" --structural-library "$STRUCTURAL_LIBRARY" \
  --output-dir "$OUTPUT_ROOT/planner-isolation"
