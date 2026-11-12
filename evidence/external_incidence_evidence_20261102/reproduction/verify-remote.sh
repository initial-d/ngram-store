set -eu
TASK_ROOT=/llm-align/duyimin/ngram-external-build-20261102
TASK_PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$TASK_ROOT/source4"
trap 'printf "failed\n" > "$TASK_ROOT/verifier-status4.txt"' ERR
"$TASK_PYTHON" -m unittest tests.test_external_cotraining -v > "$TASK_ROOT/targeted-tests4.txt" 2>&1
"$TASK_PYTHON" -m tools.verify_external_incidence \
 --build "$TASK_ROOT/results2/1855907-external-64" \
 --previous-store /llm-align/duyimin/ngram-cotraining-20261031/large-results/certified-raw-blocks \
 --bags /llm-align/duyimin/ngram-cross-corpus-20261102/large-native-bags.json \
 --expected-trace-sha256 cedaa2f0ac19ceedfea4eda07139204d8c179d84e3f33e213811168cd93f83fe \
 --native-library /llm-align/duyimin/ngram-cotraining-20261031/source/build/librows.so \
 --output "$TASK_ROOT/native-verifier4.json"
printf 'completed\n' > "$TASK_ROOT/verifier-status4.txt"
