set -eu
TASK_ROOT=/llm-align/duyimin/ngram-query-bounds-20261104
TASK_PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$TASK_ROOT/tmp"
export TMPDIR="$TASK_ROOT/tmp"
cd "$TASK_ROOT/source"
make native > "$TASK_ROOT/native-build.log" 2>&1
export MEFT_ROWS_LIBRARY="$TASK_ROOT/source/build/libmeft_rows.so"
"$TASK_PYTHON" -m tools.benchmark_budgeted_bags \
  --training-root /llm-align/duyimin/ngram-cross-corpus-20261102/training \
  --training-report docs/cross_corpus_evidence_20261102/training-report.json \
  --previous-traces-root /llm-align/duyimin/ngram-cross-corpus-20261102/complete-grid \
  --output-dir "$TASK_ROOT/results-native" \
  --source-commit da2bce4d12cfd185cfcf43540a9d442d36f9ea24 \
  > "$TASK_ROOT/benchmark-native.log" 2>&1
cat "$TASK_ROOT/benchmark-native.log"
