# Reproduce the declared CPU grid

The existing inputs are not included in this report. Recreate the cached
[trained table and provenance](../../cpu_capacity_evidence_20261030/README.md)
and [registered domain](../../cotraining_evidence_20261031/README.md), or supply
your own table/CSR and use the public construction API. Different inputs are
new experiments and must not inherit these hashes or numbers.

Use the preparation commit `25ed5e6`, build commit `1c0c832` and verifier commit
`9734ec1` listed in the report. Paths below are task-owned paths from the actual
run. Use a new output directory to avoid overwriting preserved evidence.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m tools.prepare_external_bags \
  --training-dir /llm-align/duyimin/meft-cpu-capacity-20261030/training2 \
  --words /llm-align/duyimin/ngram-cotraining-20261031/large-results/precomputed-known-words/words.json \
  --reference-report /llm-align/duyimin/ngram-cotraining-20261031/large-results/report.json \
  --output-dir /llm-align/duyimin/ngram-external-build-20261102/inputs

python -m tools.benchmark_external_incidence \
  --inputs /llm-align/duyimin/ngram-external-build-20261102/inputs \
  --output-dir /llm-align/duyimin/ngram-external-build-20261102/results2
```

The controller launches each worker with `python -m
tools.benchmark_external_incidence --worker JOB.json`. Exact job arguments are
in [jobs](../jobs/). The reports retain Python's `sys.argv` representation,
whose first argument is the resolved module file; replay with `-m` so relative
imports work. [Actual launcher](run-remote.sh) and
[separate verifier invocation](verify-remote.sh) preserve the executed commands.
The remote interpreter was `/llm-align/duyimin/miniconda3/envs/gemma4/bin/python`.

Every declared scale/budget, all byte comparisons and the successful full
native verifier are required before deriving a success summary. From a checkout
containing the finalizer (NumPy and Matplotlib needed for plotting):

```bash
python -m tools.finalize_external_incidence \
  --evidence docs/external_incidence_evidence_20261102
```

This regenerates the source-linked summary, PNG/SVG charts and manifest. Charts
use local Matplotlib 3.9.4; renderer-dependent image hashes can vary while the
numeric summary remains the same. Original runtime/corpus preparation, native
packing and OS cache costs must be reported separately.
