# Reproduce the co-training evidence

These experiments read local public-corpus caches and existing trained rows.
They do not download data or require a GPU. Reports record exact commands,
source commits, file hashes and environment; `../environment.json` supplies the
shared CPU details. Timings will vary across hosts and page-cache state.
The JSON `command` fields resolve `python -m tools.module` to the module's file
path via `sys.argv`; replay the `-m` invocations in the scripts, rather than
executing that resolved file directly (the tools use relative imports).

`rerun-small.sh` runs from a Linux repository checkout with Torch, NumPy, the
native row library and the bundled Linux DATrie bridge. Set paths explicitly:

```bash
export PY=/path/to/python
export CORPUS=/path/to/tinyshakespeare.txt
export CHECKPOINT_ROOT=/path/to/heldout-compression-20261024
export NATIVE_LIBRARY=/path/to/librows.so
export STRUCTURAL_LIBRARY=/path/to/libstructural.so
export OUTPUT_ROOT=/path/to/new-small-run
bash docs/cotraining_evidence_20261031/reproduction/rerun-small.sh
```

The checkpoint root contains `order-{4,8,12}-seed-{3,7,11}/reference.pt`
from the earlier held-out study. Corpus and checkpoint hashes must match the
primary reports for a same-source reproduction. The new training runs use
independent initialization and identical paired samples/negatives. Random group
controls redistribute actual trained rows; they are not random-vector evidence.

The large experiment additionally requires the earlier finite SGNS export:
`trained.f32`, `keys.json`, `trained-source-ids.u64`, `training.json` and its
local enwik8 ZIP. The tools check the recorded hashes before using that source.
Example commands from a checkout, with CUDA disabled and CPU thread counts one:

```bash
python -m tools.benchmark_cotraining_capacity \
  --training-dir "$TRAINING_DIR" --corpus-zip "$CORPUS_ZIP" \
  --output-dir "$TASK/large-results" --native-library "$NATIVE_LIBRARY"
python -m tools.benchmark_cotraining_layout \
  --training-dir "$TRAINING_DIR" \
  --partition-path "$TASK/large-results/certified-groups/domain.npz" \
  --output-dir "$TASK/layout-large-sequential" --native-library "$NATIVE_LIBRARY"
```

`tools.finalize_cotraining_evidence` expects primary outputs named
`cached-results`, `dynamics-legacy`, `dynamics-average` and `large-results` under
its `--task-dir`, plus cached checkpoints under
`source/artifacts/heldout-compression-20261024`. It materializes common files and
the stronger precomputed-word baseline. `tools.benchmark_cotraining_raw` follows
that audit and compares encoded/raw certified, raw mmap and raw precomputed
serving on the same trace. Its source `training.json` retains the original
`settings.corpus_zip` path; make that local corpus available when replaying.
These post-run audits are separate from the primary protocols.

The `historical/*.sh.txt` files preserve actual orchestration and verification
commands at the reported commits. They include local clone/bundle paths and a
recorded interruption of our first CPU export; they are **plain-text historical
records**, not portable launchers. Use a fresh output directory for every run.
The interrupted run supplies no final capacity result. Raw corpora, model
weights, core dumps, SSH settings and authentication material are not published.

Rebuild the figures using `python docs/cotraining_evidence_20261031/plot_evidence.py`
with Matplotlib and NumPy. The figure input is the checked-in measured JSON.
