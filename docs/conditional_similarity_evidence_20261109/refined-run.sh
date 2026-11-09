#!/bin/bash
set -eu
TASK=/llm-align/duyimin/ngram-conditional-similarity-20261109
PY=/llm-align/duyimin/ngram-counterfact-capacity-20261106/env/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$TASK/refined-source"
"$PY" -m tools.analyze_conditional_similarity checkpoints --training-root /llm-align/duyimin/ngram-cross-corpus-20261102/training --output "$TASK/refined-enwik8" --native-library "$TASK/source/build/libmeft_rows.so"
printf '%s\n' complete > "$TASK/refined-complete"
