#!/bin/bash
set -euo pipefail
TASK_ROOT=/llm-align/duyimin/ngram-news-app-20261105
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
mkdir "$TASK_ROOT/source"
tar -xzf "$TASK_ROOT/source.tar.gz" -C "$TASK_ROOT/source"
cd "$TASK_ROOT/source"
make native > "$TASK_ROOT/native-build.log" 2>&1
export MEFT_ROWS_LIBRARY="$TASK_ROOT/source/build/libmeft_rows.so"
/llm-align/duyimin/miniconda3/envs/gemma4/bin/python -m unittest tests.test_text_classifier -v > "$TASK_ROOT/remote-targeted.log" 2>&1
/llm-align/duyimin/miniconda3/envs/gemma4/bin/python -m tools.train_news_classifier \
  --data "$TASK_ROOT/data" --output "$TASK_ROOT/training" --source-commit 698efcea8210eb9ee65bd278dde3a0ab214d7f62 \
  > "$TASK_ROOT/training.log" 2>&1
tail -n 8 "$TASK_ROOT/training.log"
