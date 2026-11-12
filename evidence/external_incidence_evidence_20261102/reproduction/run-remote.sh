set -eu
TASK_ROOT=/llm-align/duyimin/ngram-external-build-20261102
TASK_PYTHON=/llm-align/duyimin/miniconda3/envs/gemma4/bin/python
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$TASK_ROOT/tmp"
cd "$TASK_ROOT/source2"
trap 'printf "failed\n" > "$TASK_ROOT/status2.txt"' ERR
"$TASK_PYTHON" -m unittest tests.test_external_cotraining tests.test_cotraining_groups -v > "$TASK_ROOT/targeted-tests2.txt" 2>&1
"$TASK_PYTHON" -m tools.benchmark_external_incidence --inputs "$TASK_ROOT/inputs" --output-dir "$TASK_ROOT/results2"
"$TASK_PYTHON" -m unittest discover -s tests > "$TASK_ROOT/full-tests2.txt" 2>&1
printf 'completed\n' > "$TASK_ROOT/status2.txt"
