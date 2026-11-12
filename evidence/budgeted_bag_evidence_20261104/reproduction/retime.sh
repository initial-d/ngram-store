set -eu
TASK_ROOT=/llm-align/duyimin/ngram-query-bounds-20261104
TASK_PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
export MEFT_ROWS_LIBRARY="$TASK_ROOT/source/build/libmeft_rows.so"
cd "$TASK_ROOT/batched-r2-source"
"$TASK_PYTHON" -m tools.retime_budgeted_bags \
  --report "$TASK_ROOT/results-native/report.json" \
  --results-root "$TASK_ROOT/results-native" \
  --previous-traces-root /llm-align/duyimin/ngram-cross-corpus-20261102/complete-grid \
  --reference-module "$TASK_ROOT/source/tools/budgeted_bags.py" \
  --source-commit e55927279735a12759163a3c762e16e74b96772a \
  --output "$TASK_ROOT/batched-r2-retiming.json" \
  > "$TASK_ROOT/batched-r2-retiming.log" 2>&1
"$TASK_PYTHON" -m tools.analyze_budgeted_bags \
  --report "$TASK_ROOT/results-native/report.json" \
  --retiming "$TASK_ROOT/batched-r2-retiming.json" \
  --output-dir "$TASK_ROOT/public-summary" --plots \
  > "$TASK_ROOT/analysis.log" 2>&1
cat "$TASK_ROOT/batched-r2-retiming.log"
cat "$TASK_ROOT/analysis.log"
