set -eu
GRID=/llm-align/duyimin/ngram-qwen-gpu-grid-20261106
OLD=/llm-align/duyimin/ngram-qwen-identity-gpu-20261106
PY=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
cd "$GRID/source"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export MEFT_TINYENGRAM_SOURCE=/llm-align/duyimin/meft-cuda-20261030/upstream
export CUDA_VISIBLE_DEVICES=GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8
"$PY" -m unittest tests.test_tinyengram_checkpoint.TinyEngramCheckpointTest.test_cpu_identity_training_exports_native_and_evaluates_complete_splits tests.test_tinyengram_checkpoint.TinyEngramCheckpointTest.test_cuda_identity_training_exports_native_and_evaluates_complete_splits -v > "$GRID/training-resume-tests.log" 2>&1
run_one() {
  gpu_uuid=$1
  order=$2
  seed=$3
  mode=$4
  output="$GRID/order$order-seed$seed"
  protocol=docs/qwen_identity_gpu_protocol_20261106.json
  if test "$order" != 4; then protocol="docs/qwen_identity_gpu_order${order}_protocol_20261106.json"; fi
  rc=0
  env CUDA_VISIBLE_DEVICES="$gpu_uuid" "$PY" -m tools.train_tinyengram_identity --device cuda --seed "$seed" \
    --base-model /llm-align/open_models/Qwen3/Qwen3-0.6B \
    --corpus /llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt \
    --upstream-dir /llm-align/duyimin/meft-cuda-20261030/upstream \
    --native-library "$GRID/libmeft_rows.so" --protocol "$protocol" \
    "$mode" "$OLD/training" --output-dir "$output" > "$output.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$output.exit-code"
  return "$rc"
}
(run_one GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8 4 3 --resume-from && run_one GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8 3 11 --prefix-from) &
pids="$!"
for entry in \
 'GPU-dd9f15a3-3b6d-8e11-3ed4-c7ab98a57079 4 7' \
 'GPU-805ec4aa-37cf-6d18-f341-da44657c3844 4 11' \
 'GPU-7c1fe9b0-4b49-e530-ebab-5cb72ec71aa4 6 3' \
 'GPU-739011e2-6861-76a1-aacc-ec382e95e733 6 7' \
 'GPU-bc11cba9-e025-335f-83a8-9ec15f87a496 6 11' \
 'GPU-20d75cd9-0182-957b-65df-9a2d44291c6e 3 3' \
 'GPU-0e8947d0-b90d-8e5a-06dd-342ef5db3cc8 3 7'; do
  read -r uuid order seed <<< "$entry"
  run_one "$uuid" "$order" "$seed" --prefix-from &
  pids="$pids $!"
 done
for pid in $pids; do wait "$pid"; done
printf 'passed\n' > "$GRID/status.txt"
