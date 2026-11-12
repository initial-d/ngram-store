#!/bin/bash
set -euo pipefail
TASK_ROOT=/llm-align/duyimin/ngram-news-app-20261105
PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
mkdir "$TASK_ROOT/rss-source"
tar -xzf "$TASK_ROOT/rss-source.tar.gz" -C "$TASK_ROOT/rss-source"
cd "$TASK_ROOT/rss-source"
make native > "$TASK_ROOT/rss-native-build.log" 2>&1
export MEFT_ROWS_LIBRARY="$TASK_ROOT/rss-source/build/libmeft_rows.so"
"$PYTHON" -m unittest tests.test_text_classifier -v > "$TASK_ROOT/rss-targeted.log" 2>&1
# Correct only the confirmed RSS measurement bug. No retraining, model,
# vocabulary, codec parameters, quality gates or budget changes.
for task in relu linear; do
  if [ "$task" = linear ]; then TRAINING="$TASK_ROOT/training"; else TRAINING="$TASK_ROOT/training-relu"; fi
  "$PYTHON" -m tools.benchmark_news_classifier prepare --data "$TASK_ROOT/data" \
    --training "$TRAINING" --output "$TASK_ROOT/requests-rss/$task"
  for seed in 3 7 11; do
    if [ "$task" = linear ]; then
      SERVING="$TASK_ROOT/serving-final/linear/seed-$seed"
    else
      SERVING="$TASK_ROOT/serving-rss/relu/seed-$seed"
      "$PYTHON" -m tools.export_news_classifier --training "$TRAINING" \
        --output "$SERVING" --seed "$seed" --native-library "$MEFT_ROWS_LIBRARY" \
        > "$TASK_ROOT/export-rss-$task-$seed.log" 2>&1
    fi
    "$PYTHON" -m tools.benchmark_news_classifier run --serving "$SERVING" \
      --requests "$TASK_ROOT/requests-rss/$task" --output "$TASK_ROOT/results-rss/$task/seed-$seed" --seed "$seed" \
      --source-commit 237cb86953a4a0061e1047b5a82d4c8d2d3d00cc --native-library "$MEFT_ROWS_LIBRARY" \
      > "$TASK_ROOT/benchmark-rss-$task-$seed.log" 2>&1
  done
done
tail -n 12 "$TASK_ROOT/benchmark-rss-linear-11.log"
