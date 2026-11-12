#!/bin/bash
set -eu
TASK=${RUN_ROOT}
PY=${ENV_ROOT}/bin/python
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export MEFT_SOURCE_COMMIT=b58c3fa31ce73cb2dff36963a98f4b2a9e9503b4
cd "$TASK/frontier-source"
make native structural-native > "$TASK/frontier-build.log" 2>&1
"$PY" -m tools.tree_node_quality prepare --training-root ${TRAINING_ROOT} --corpus-zip ${CORPUS_ZIP} --output "$TASK/frontier" --native-library "$TASK/frontier-source/build/libmeft_rows.so" --structural-library "$TASK/frontier-source/build/libmeft_structural.so" > "$TASK/frontier-prepare.log" 2>&1
printf '%s\n' prepared > "$TASK/frontier-prepared"
