#!/bin/bash
set -eu
TASK=${RUN_ROOT}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export MEFT_SOURCE_COMMIT=fb4e30bb75f66b0b641012a8038932d6711ece87
cd "$TASK/source"
make native structural-native > "$TASK/build.log" 2>&1
${ENV_ROOT}/bin/python -m tools.train_tree_node_mechanism --training-root ${TRAINING_ROOT} --output "$TASK/mechanism" --jobs 9 > "$TASK/mechanism.log" 2>&1
printf '%s\n' complete > "$TASK/mechanism-complete"
