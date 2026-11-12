#!/bin/bash
set -eu
ROOT=/llm-align/duyimin/ngram-qwen-joint-20261106
PY=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
cd "$ROOT/source"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MEFT_TINYENGRAM_SOURCE=/llm-align/duyimin/meft-cuda-20261030/upstream
run_control() {
 uuid=$1
 seed=$2
 control=$3
 output="$ROOT/$control-seed$seed"
 args=()
 if test "$control" != full; then args=(--prefix-from "$ROOT/full-seed$seed"); fi
 rc=0
 env CUDA_VISIBLE_DEVICES="$uuid" "$PY" -m tools.train_tinyengram_joint --device cuda --seed "$seed" --control "$control" \
  --base-model /llm-align/open_models/Qwen3/Qwen3-0.6B \
  --corpus /llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt \
  --upstream-dir /llm-align/duyimin/meft-cuda-20261030/upstream \
  --native-library "$ROOT/libmeft_rows.so" --protocol docs/qwen_joint_gpu_protocol_20261106.json \
  "${args[@]}" --output-dir "$output" > "$output.log" 2>&1 || rc=$?
 printf '%s\n' "$rc" > "$output.exit-code"
 return "$rc"
}
run_seed() {
 run_control "$1" "$2" full
 run_control "$1" "$2" head-constant
 run_control "$1" "$2" off
}
run_seed GPU-20d75cd9-0182-957b-65df-9a2d44291c6e 3 &
a=$!
run_seed GPU-0e8947d0-b90d-8e5a-06dd-342ef5db3cc8 7 &
b=$!
run_seed GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8 11 &
c=$!
rc=0
for p in "$a" "$b" "$c"; do wait "$p" || rc=$?; done
if test "$rc" = 0; then
 "$PY" -m tools.analyze_tinyengram_joint --training-root "$ROOT" --protocol docs/qwen_joint_gpu_protocol_20261106.json --output "$ROOT/summary.json" || rc=$?
fi
printf '%s\n' "$rc" > "$ROOT/experiment.exit-code"
touch "$ROOT/experiment-finished"
exit "$rc"
