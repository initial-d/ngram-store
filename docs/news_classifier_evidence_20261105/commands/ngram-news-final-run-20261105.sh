#!/bin/bash
set -euo pipefail
TASK_ROOT=/llm-align/duyimin/ngram-news-app-20261105
PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
mkdir "$TASK_ROOT/final-source"
tar -xzf "$TASK_ROOT/final-source.tar.gz" -C "$TASK_ROOT/final-source"
cd "$TASK_ROOT/final-source"
make native > "$TASK_ROOT/final-native-build.log" 2>&1
export MEFT_ROWS_LIBRARY="$TASK_ROOT/final-source/build/libmeft_rows.so"
"$PYTHON" -m unittest tests.test_text_classifier -v > "$TASK_ROOT/final-targeted.log" 2>&1
"$PYTHON" -m tools.train_news_classifier --data "$TASK_ROOT/data" \
  --output "$TASK_ROOT/training-relu" --activation relu --source-commit 84c21eae7bbef4c1ee505b1f7336ef48b5d80ac8 \
  > "$TASK_ROOT/training-relu.log" 2>&1
for task in linear relu; do
  if [ "$task" = linear ]; then TRAINING="$TASK_ROOT/training"; else TRAINING="$TASK_ROOT/training-relu"; fi
  "$PYTHON" -m tools.benchmark_news_classifier prepare --data "$TASK_ROOT/data" \
    --training "$TRAINING" --output "$TASK_ROOT/requests-final/$task"
  for seed in 3 7 11; do
    "$PYTHON" -m tools.export_news_classifier --training "$TRAINING" \
      --output "$TASK_ROOT/serving-final/$task/seed-$seed" --seed "$seed" \
      --native-library "$MEFT_ROWS_LIBRARY" > "$TASK_ROOT/export-final-$task-$seed.log" 2>&1
    "$PYTHON" -m tools.benchmark_news_classifier run --serving "$TASK_ROOT/serving-final/$task/seed-$seed" \
      --requests "$TASK_ROOT/requests-final/$task" --output "$TASK_ROOT/results-final/$task/seed-$seed" --seed "$seed" \
      --source-commit 84c21eae7bbef4c1ee505b1f7336ef48b5d80ac8 --native-library "$MEFT_ROWS_LIBRARY" \
      > "$TASK_ROOT/benchmark-final-$task-$seed.log" 2>&1
  done
done
tail -n 11 "$TASK_ROOT/benchmark-final-relu-11.log"
