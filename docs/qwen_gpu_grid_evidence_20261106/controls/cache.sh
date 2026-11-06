set -u
ROOT=/llm-align/duyimin/ngram-qwen-codecs-20261106
GRID=/llm-align/duyimin/ngram-qwen-gpu-grid-20261106
PY=/llm-align/duyimin/meft-cuda-20261030/compat/bin/python
cd "$ROOT/cache-source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" "$ROOT/release-cache-occupiers.py" &
releaser=$!
finish(){ touch "$ROOT/cache-finished";wait "$releaser"; }
trap finish EXIT
one(){
 uuid=$1; seed=$2
 output="$ROOT/cache-serving/seed$seed"
 mkdir -p "$output"
 for entry in 'resident blocks resident' 'native blocks native-blocks' 'native rows native-rows';do
  read -r mode policy name <<< "$entry"
  success=0
  for attempt in 1 2 3; do
   rc=0
   env CUDA_VISIBLE_DEVICES="$uuid" "$PY" -m tools.benchmark_tinyengram_cuda \
    --checkpoint "$GRID/order4-seed$seed/seed-$seed/checkpoint" --bundle "$GRID/order4-seed$seed/seed-$seed/bundle" \
    --corpus /llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt --upstream-dir /llm-align/duyimin/meft-cuda-20261030/upstream \
    --native-library "$GRID/libmeft_rows.so" --output-dir "$output" --mode "$mode" --report-name "$name" \
    --native-cache-policy "$policy" --tokens 32,128 --decode-tokens 16 --staging-rows 4096 --warmup 2 --repeat 7 \
    > "$output/$name-attempt$attempt.log" 2>&1 || rc=$?
   printf '%s\n' "$rc" > "$output/$name-attempt$attempt.exit-code"
   if test "$rc" = 0; then success=1;break;fi
   if test -f "$output/$name.json";then mv "$output/$name.json" "$output/$name-attempt$attempt.json";fi
   sleep 3
  done
  if test "$success" != 1;then printf 'failed %s\n' "$name" > "$output/status.txt";return 1;fi
 done
 printf 'passed\n' > "$output/status.txt"
}
pids=''
for entry in \
 'GPU-20d75cd9-0182-957b-65df-9a2d44291c6e 3' \
 'GPU-0e8947d0-b90d-8e5a-06dd-342ef5db3cc8 7' \
 'GPU-0d6f5ec3-1a8c-b91d-cab5-0cf73accebf8 11';do
 read -r uuid seed <<< "$entry"
 one "$uuid" "$seed" &
 pids="$pids $!"
done
for pid in $pids;do wait "$pid";done
printf 'finished; inspect each status\n' > "$ROOT/cache-status.txt"
