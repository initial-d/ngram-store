set -eu
TASK_ROOT=/llm-align/duyimin/ngram-cross-corpus-20261102
TASK_PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
cd "$TASK_ROOT/grid-source"
trap 'printf "failed\n" > "$TASK_ROOT/grid-status.txt"' ERR
"$TASK_PYTHON" -m tools.benchmark_faiss_checkpoint_grid --checkpoint-root "$TASK_ROOT/training/legacy" \
 --corpus "$TASK_ROOT/training/corpus.txt" --output-dir "$TASK_ROOT/complete-grid" \
 --reuse-root "$TASK_ROOT/faiss-results" --jobs 3 --train-threads 4 --orders 4,8,12 --seeds 3,7,11
"$TASK_PYTHON" -m tools.benchmark_faiss_code_dedup --roots "$TASK_ROOT/complete-grid" --output-dir "$TASK_ROOT/complete-code-dedup"
"$TASK_PYTHON" -m tools.analyze_faiss_cotraining --report "$TASK_ROOT/complete-grid/report.json" \
 --dedup-report "$TASK_ROOT/complete-code-dedup/report.json" --raw-root "$TASK_ROOT/complete-grid" \
 --dedup-root "$TASK_ROOT/complete-code-dedup" --output-dir "$TASK_ROOT/complete-analysis" --plots
"$TASK_PYTHON" -m tools.benchmark_faiss_schedule --report "$TASK_ROOT/complete-analysis/merged-report.json.gz" \
 --roots "$TASK_ROOT/complete-grid" "$TASK_ROOT/complete-code-dedup" \
 --reference-module "$TASK_ROOT/reference-faiss.py" --output-dir "$TASK_ROOT/complete-cross-retiming" \
 --loss-budget 0.001 --repeat 15 --warmup 2
printf 'completed\n' > "$TASK_ROOT/grid-status.txt"
