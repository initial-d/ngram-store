set -u
ROOT=/llm-align/duyimin/ngram-qwen-codecs-20261106
GRID=/llm-align/duyimin/ngram-qwen-gpu-grid-20261106
PY=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
cd "$ROOT/source"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export MEFT_TINYENGRAM_SOURCE=/llm-align/duyimin/meft-cuda-20261030/upstream
"$PY" "$ROOT/release-floor.py" &
releaser=$!
finish(){ touch "$ROOT/all-finished"; wait "$releaser"; }
trap finish EXIT
run_one(){
 uuid=$1; order=$2; seed=$3
 output="$ROOT/order$order-seed$seed"
 rc=0
 env CUDA_VISIBLE_DEVICES="$uuid" "$PY" -m tools.evaluate_tinyengram_codecs \
  --training-run "$GRID/order$order-seed$seed/report.json" --seed-directory "$GRID/order$order-seed$seed/seed-$seed" \
  --prefix-root /llm-align/duyimin/ngram-qwen-identity-gpu-20261106/training \
  --corpus /llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt --upstream-dir "$MEFT_TINYENGRAM_SOURCE" \
  --native-library "$GRID/libmeft_rows.so" --output-dir "$output" > "$output.log" 2>&1 || rc=$?
 printf '%s\n' "$rc" > "$output.exit-code"
 return "$rc"
}
low_lr(){
 uuid=$1; seed=$2; output="$ROOT/low-lr-seed$seed"; rc=0
 env CUDA_VISIBLE_DEVICES="$uuid" "$PY" -m tools.train_tinyengram_identity --device cuda --seed "$seed" \
  --base-model /llm-align/open_models/Qwen3/Qwen3-0.6B --corpus /llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt \
  --upstream-dir "$MEFT_TINYENGRAM_SOURCE" --native-library "$GRID/libmeft_rows.so" \
  --protocol docs/qwen_identity_gpu_low_lr_protocol_20261106.json --output-dir "$output" > "$output.log" 2>&1 || rc=$?
 printf '%s\n' "$rc" > "$output.exit-code"
 return "$rc"
}
pids=''
for entry in \
 'GPU-dd9f15a3-3b6d-8e11-3ed4-c7ab98a57079 4 3' \
 'GPU-805ec4aa-37cf-6d18-f341-da44657c3844 4 7' \
 'GPU-7c1fe9b0-4b49-e530-ebab-5cb72ec71aa4 4 11' \
 'GPU-739011e2-6861-76a1-aacc-ec382e95e733 6 3' \
 'GPU-bc11cba9-e025-335f-83a8-9ec15f87a496 6 7' \
 'GPU-20d75cd9-0182-957b-65df-9a2d44291c6e 6 11' \
 'GPU-0e8947d0-b90d-8e5a-06dd-342ef5db3cc8 3 3' \
 'GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8 3 7'; do
 read -r uuid order seed <<< "$entry"
 (run_one "$uuid" "$order" "$seed"
  if test "$uuid" = GPU-dd9f15a3-3b6d-8e11-3ed4-c7ab98a57079; then run_one "$uuid" 3 11; fi
  if test "$uuid" = GPU-20d75cd9-0182-957b-65df-9a2d44291c6e; then low_lr "$uuid" 3; fi
  if test "$uuid" = GPU-0e8947d0-b90d-8e5a-06dd-342ef5db3cc8; then low_lr "$uuid" 7; fi
  if test "$uuid" = GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8; then low_lr "$uuid" 11; fi
 ) &
 pids="$pids $!"
done
for pid in $pids; do wait "$pid"; done
printf 'finished; inspect every shard exit code\n' > "$ROOT/status.txt"
