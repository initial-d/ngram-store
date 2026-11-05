#!/bin/bash
set -euo pipefail
TASK_ROOT=/llm-align/duyimin/ngram-native-bounds-20261105
PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
mkdir "$TASK_ROOT/source" "$TMPDIR"
tar -xzf "$TASK_ROOT/source.tar.gz" -C "$TASK_ROOT/source"
cd "$TASK_ROOT/source"
make native > "$TASK_ROOT/native-build.log" 2>&1
export MEFT_ROWS_LIBRARY="$TASK_ROOT/source/build/libmeft_rows.so"
"$PYTHON" -m unittest tests.test_budgeted_bags -v > "$TASK_ROOT/remote-targeted.log" 2>&1
"$PYTHON" tools/check_native_sanitizers.py > "$TASK_ROOT/remote-sanitizers.log" 2>&1
"$PYTHON" -m tools.benchmark_native_bounded_bags \
  --report /llm-align/duyimin/ngram-query-bounds-20261104/results-native/report.json \
  --prior-retiming /llm-align/duyimin/ngram-query-bounds-20261104/batched-r2-retiming.json \
  --results-root /llm-align/duyimin/ngram-query-bounds-20261104/results-native \
  --traces-root /llm-align/duyimin/ngram-cross-corpus-20261102/complete-grid \
  --output "$TASK_ROOT/report.json" \
  --source-commit 56df5865dd58ae5a396ffa2910b8c953e9511782 \
  > "$TASK_ROOT/benchmark.log" 2>&1
cat "$TASK_ROOT/benchmark.log"
