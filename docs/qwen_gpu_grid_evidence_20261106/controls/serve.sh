set -u
ROOT=/llm-align/duyimin/ngram-qwen-codecs-20261106
GRID=/llm-align/duyimin/ngram-qwen-gpu-grid-20261106
PY=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
cd "$ROOT/perf-source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" "$ROOT/release-occupiers.py" &
releaser=$!
finish(){ touch "$ROOT/serving-finished"; wait "$releaser"; }
trap finish EXIT
one(){
 uuid=$1; order=$2; seed=$3
 output="$ROOT/serving/order$order-seed$seed"
 mkdir -p "$output"
 for entry in resident dense-offload native fp16/raw fp16/native int8/raw int8/native sq8 sq4 pq-8x8 cluster-8 opq-8x8; do
  mode="$entry"; name="$entry"; codec=''
  case "$entry" in
   resident|dense-offload|native) ;;
   *) mode=codec; name="codec-${entry//\//-}"; codec="$ROOT/order$order-seed$seed/$entry" ;;
  esac
  success=0
  for attempt in 1 2 3; do
   rc=0
   extra=()
   if test -n "$codec"; then extra=(--codec-directory "$codec"); fi
   env CUDA_VISIBLE_DEVICES="$uuid" "$PY" -m tools.benchmark_tinyengram_cuda \
    --checkpoint "$GRID/order$order-seed$seed/seed-$seed/checkpoint" --bundle "$GRID/order$order-seed$seed/seed-$seed/bundle" \
    --corpus /llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt --upstream-dir /llm-align/duyimin/meft-cuda-20261030/upstream \
    --native-library "$GRID/libmeft_rows.so" --output-dir "$output" --mode "$mode" --report-name "$name" \
    --tokens 32,128 --decode-tokens 16 --staging-rows 4096 --warmup 2 --repeat 7 "${extra[@]}" > "$output/$name-attempt$attempt.log" 2>&1 || rc=$?
   printf '%s\n' "$rc" > "$output/$name-attempt$attempt.exit-code"
   if test "$rc" = 0; then success=1;break;fi
   if test -f "$output/$name.json"; then mv "$output/$name.json" "$output/$name-attempt$attempt.json"; fi
   sleep 3
  done
  if test "$success" != 1; then printf 'failed %s\n' "$name" > "$output/status.txt";return 1;fi
 done
 printf 'passed\n' > "$output/status.txt"
}
pids=''
for entry in \
 'GPU-805ec4aa-37cf-6d18-f341-da44657c3844 4 3' \
 'GPU-7c1fe9b0-4b49-e530-ebab-5cb72ec71aa4 4 7' \
 'GPU-739011e2-6861-76a1-aacc-ec382e95e733 4 11' \
 'GPU-bc11cba9-e025-335f-83a8-9ec15f87a496 6 3'; do
 read -r uuid order seed <<< "$entry"
 (one "$uuid" "$order" "$seed") &
 pids="$pids $!"
done
# GPU 0 first finishes the queued order-3 quality job; never stop it.
while ps -eo args | awk '/tools.evaluate_tinyengram_codecs/ && /order3-seed11/ && !/awk/ {found=1} END {exit !found}'; do sleep 3;done
one GPU-dd9f15a3-3b6d-8e11-3ed4-c7ab98a57079 3 3 &
pids="$pids $!"
for pid in $pids;do wait "$pid";done
printf 'finished; inspect shard statuses\n' > "$ROOT/serving-status.txt"
