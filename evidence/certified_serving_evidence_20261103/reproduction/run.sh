set -eu
TASK_ROOT=/llm-align/duyimin/ngram-store-serving-20261103
TASK_PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$TASK_ROOT/source"
trap 'printf "failed\n" > "$TASK_ROOT/status.txt"' ERR
"$TASK_PYTHON" -m unittest tests.test_certified_store tests.test_certified_torch -v > "$TASK_ROOT/targeted-tests.txt" 2>&1
"$TASK_PYTHON" -m tools.benchmark_certified_serving \
 --input /llm-align/duyimin/ngram-external-build-20261102/inputs/1855907/input.json \
 --output-dir "$TASK_ROOT/results"
"$TASK_PYTHON" -m tools.benchmark_certified_serving --verify \
 --input /llm-align/duyimin/ngram-external-build-20261102/inputs/1855907/input.json \
 --store "$TASK_ROOT/results/external" \
 --previous /llm-align/duyimin/ngram-cotraining-20261031/large-results/certified-raw-blocks \
 --bags /llm-align/duyimin/ngram-cross-corpus-20261102/large-native-bags.json \
 --native-library "$TASK_ROOT/librows.so" --output "$TASK_ROOT/verifier.json"
"$TASK_PYTHON" -m ngram_store.cli demo-bags --output "$TASK_ROOT/demo" --native-library "$TASK_ROOT/librows.so" > "$TASK_ROOT/demo.log" 2>&1
"$TASK_PYTHON" -m unittest discover -s tests > "$TASK_ROOT/full-tests.txt" 2>&1
printf 'completed\n' > "$TASK_ROOT/status.txt"
