#!/bin/bash
set -eu
TASK=/llm-align/duyimin/ngram-implicit-format-20261109
BASE=/llm-align/duyimin/ngram-native-real-20261109
PY=/llm-align/duyimin/ngram-counterfact-capacity-20261106/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$TASK/source"
for model in qwen06 qwen35; do
  for seed in 3 7 11; do
    tag="$model-20877-exact-s$seed"
    blocks=32
    if [ "$seed" = 3 ]; then blocks=32,128,512,2048; fi
    "$PY" -m tools.benchmark_identity_format --fixture "$BASE/results/$tag" --output "$TASK/results/$tag" --native-library "$TASK/source/build/libmeft_rows.so" --old-library "$BASE/source/build/libmeft_rows.so" --block-sizes "$blocks" --cache-sizes 0,1048576 --warmup 3 --repeat 20
  done
done
for model in qwen06 qwen35; do
  tag="$model-20877-exact-s3"
  for mode in dense raw-mmap; do
    "$PY" -m tools.benchmark_counterfact_storage worker --fixture "$TASK/results/$tag" --native-library "$TASK/source/build/libmeft_rows.so" --mode "$mode" --output "$TASK/results/$tag/$mode.json" --cache-bytes 0 --warmup 3 --repeat 20
  done
done
printf '%s\n' complete > "$TASK/complete"
