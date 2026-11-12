set -eu
ROOT=/llm-align/duyimin/ngram-counterfact-capacity-20261106
trap 'printf "[1,2,3,5,6,7]\n" > "$ROOT/release-scope.json"' EXIT
cd "$ROOT/source-native"
export PYTHONPATH="$ROOT/source-native" CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=GPU-bc11cba9-e025-335f-83a8-9ec15f87a496
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HOME="$ROOT/cache/hf" TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" XDG_CACHE_HOME="$ROOT/cache"
"$ROOT/env/bin/python" - <<'PY'
from pathlib import Path
import time
r=Path('/llm-align/duyimin/ngram-counterfact-capacity-20261106');deadline=time.monotonic()+7200
while not (r/'probe-finished').exists():
 if time.monotonic()>deadline:raise TimeoutError('probe not complete; native check not started')
 time.sleep(5)
PY
for model in qwen06 qwen35; do
 if [ "$model" = qwen06 ]; then BASE=/llm-align/open_models/Qwen3/Qwen3-0.6B; else BASE=/llm-align/open_models/Qwen3.5/Qwen3.5-4B; fi
 "$ROOT/env/bin/python" -m tools.counterfact_native --run "$ROOT/probe-runs/$model-1024-large-s3" --base "$BASE" --upstream /llm-align/duyimin/meft-cuda-20261030/upstream --prepared "$ROOT/prepared-$model-1024" --output "$ROOT/native-$model-probe" --native-library "$ROOT/source-native/build/libmeft_rows.so" --requests 16 > "$ROOT/native-$model-probe.log" 2>&1
done
printf '%s\n' complete > "$ROOT/native-finished"
