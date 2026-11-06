set -eu
ROOT=/llm-align/duyimin/ngram-counterfact-capacity-20261106
cd "$ROOT/source-probe"
export PYTHONPATH="$ROOT/source-probe" CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=GPU-bc11cba9-e025-335f-83a8-9ec15f87a496
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HOME="$ROOT/cache/hf" TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" XDG_CACHE_HOME="$ROOT/cache"
"$ROOT/env/bin/python" -m unittest tests.test_counterfact_data tests.test_counterfact_retrieval tests.test_late_memory_replay -v > "$ROOT/probe-fixtures.log" 2>&1
for model in qwen06 qwen35; do
 if [ "$model" = qwen06 ]; then BASE=/llm-align/open_models/Qwen3/Qwen3-0.6B; else BASE=/llm-align/open_models/Qwen3.5/Qwen3.5-4B; fi
 for variant in large adapter; do
  "$ROOT/env/bin/python" -m tools.counterfact_capacity train --base "$BASE" --upstream /llm-align/duyimin/meft-cuda-20261030/upstream --prepared "$ROOT/prepared-$model-1024" --cache "$ROOT/cache-$model-1024" --output "$ROOT/probe-runs/$model-1024-$variant-s3" --protocol "$ROOT/source-probe/benchmarks/counterfact_memory_only_probe_v1.json" --seed 3 --variant "$variant" > "$ROOT/probe-$model-$variant.log" 2>&1
 done
done
printf '%s\n' complete > "$ROOT/probe-finished"
