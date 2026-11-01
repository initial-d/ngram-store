#!/usr/bin/env bash
set -euo pipefail
: "${PY:?Set the Python executable}"
: "${CHECKPOINT_ROOT:?Set the cached nine-checkpoint root}"
: "${CORPUS:?Set the cached TinyShakespeare text}"
: "${OUTPUT_ROOT:?Set a new output root}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$OUTPUT_ROOT"
"$PY" -m tools.benchmark_faiss_cotraining --checkpoint-root "$CHECKPOINT_ROOT" --corpus "$CORPUS" \
 --output-dir "$OUTPUT_ROOT/base" --codecs fp32,sq-fp16,sq8,cluster-8,pq-8x4,pq-8x8,pq-16x8,opq-8x8,rq-2x8 \
 --train-threads 1 --orders 4,8,12 --seeds 3,7,11
"$PY" -m tools.benchmark_faiss_cotraining --checkpoint-root "$CHECKPOINT_ROOT" --corpus "$CORPUS" \
 --output-dir "$OUTPUT_ROOT/rates" --codecs sq4,sq6,pq-32x8,pq-64x8,opq-32x8,opq-64x8,rq-4x8 \
 --train-threads 4 --orders 4,8,12 --seeds 3,7,11
"$PY" -m tools.benchmark_faiss_cotraining --checkpoint-root "$CHECKPOINT_ROOT" --corpus "$CORPUS" \
 --output-dir "$OUTPUT_ROOT/clusters" --codecs cluster-9,cluster-10,cluster-11 \
 --train-threads 4 --orders 4,8,12 --seeds 3,7,11
"$PY" -m tools.benchmark_faiss_code_dedup --roots "$OUTPUT_ROOT/base" "$OUTPUT_ROOT/rates" "$OUTPUT_ROOT/clusters" \
 --output-dir "$OUTPUT_ROOT/dedup"
"$PY" -m tools.analyze_faiss_cotraining --report "$OUTPUT_ROOT/base/report.json" --supplement-report "$OUTPUT_ROOT/rates/report.json" \
 --cluster-report "$OUTPUT_ROOT/clusters/report.json" --dedup-report "$OUTPUT_ROOT/dedup/report.json" \
 --raw-root "$OUTPUT_ROOT/base" --supplement-root "$OUTPUT_ROOT/rates" --cluster-root "$OUTPUT_ROOT/clusters" \
 --dedup-root "$OUTPUT_ROOT/dedup" --output-dir "$OUTPUT_ROOT/analysis" --plots
