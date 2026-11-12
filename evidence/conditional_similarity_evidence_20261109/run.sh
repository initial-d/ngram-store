#!/bin/bash
set -eu
TASK=/llm-align/duyimin/ngram-conditional-similarity-20261109
PY=/llm-align/duyimin/ngram-counterfact-capacity-20261106/env/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$TASK/source"
"$PY" -m tools.analyze_conditional_similarity checkpoints --training-root /llm-align/duyimin/ngram-cross-corpus-20261102/training --output "$TASK/enwik8" --native-library "$TASK/source/build/libmeft_rows.so"
"$PY" -m tools.analyze_conditional_similarity qwen --training-root /llm-align/duyimin/ngram-native-real-20261109/results --output "$TASK/qwen" --upstream /llm-align/duyimin/ngram-native-real-20261109/upstream
printf '%s\n' complete > "$TASK/complete"
