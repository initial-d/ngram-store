set -eu
ROOT=/llm-align/duyimin/ngram-qwen-codecs-20261106
GRID=/llm-align/duyimin/ngram-qwen-gpu-grid-20261106
cd "$ROOT/relation-source"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for order in 3 4 6; do
 for seed in 3 7 11; do
 /llm-align/duyimin/meft-cuda-20261030/compat/bin/python -m tools.analyze_tinyengram_relations \
  --training-run "$GRID/order$order-seed$seed/report.json" --seed-directory "$GRID/order$order-seed$seed/seed-$seed" \
  --corpus /llm-align/duyimin/meft-cuda-20261030/delta/corpus.txt --upstream-dir /llm-align/duyimin/meft-cuda-20261030/upstream \
  --report-out "$ROOT/relations/order$order-seed$seed.json" > "$ROOT/relation-order$order-seed$seed.log" 2>&1
 done
done
printf 'passed\n' > "$ROOT/relation-status.txt"
