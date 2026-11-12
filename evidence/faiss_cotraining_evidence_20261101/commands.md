# Reproduction commands and source boundaries

Run from the NGramStore repository with an environment containing the versions
in [phase-provenance.json](phase-provenance.json). The original cached
checkpoints must match the report's SHA256 values. These commands neither
download data nor start GPU work. Use new output directories.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=/absolute/path/to/python
export CHECKPOINT_ROOT=/absolute/path/to/nine-cached-checkpoints
export CORPUS=/absolute/path/to/tinyshakespeare.txt
export OUTPUT_ROOT=/absolute/path/to/new-output
```

## Recorded model phases

The following shared options were explicitly used for all three recorded
phases; all unspecified options are frozen by each source commit:

```bash
"$PY" -m tools.benchmark_faiss_cotraining \
  --checkpoint-root "$CHECKPOINT_ROOT" --corpus "$CORPUS" \
  --output-dir "$PHASE_OUTPUT" --orders 4,8,12 --seeds 3,7,11 \
  --train-rows 16384 --iterations 20 --opq-iterations 10 \
  --query-bags 512 --batch-bags 64 --eval-pairs 8192 --negatives 5 \
  --repeat 15 --loss-budgets 0.0001,0.001,0.005 \
  --max-discrimination-drop 0.005 --max-aggregate-error 0.05
```

Use separate checkouts/worktrees at the source commits in the README. For the
base `a068a714` phase, use its default nine-codec grid and **do not add
`--train-threads`**, which this older source does not support. Set
`PHASE_OUTPUT` to the new base directory.

At `64ebc742`, add the following options for the rate phase:

```text
--train-threads 4 --codecs sq4,sq6,pq-32x8,pq-64x8,opq-32x8,opq-64x8,rq-4x8
```

At `3147a21d`, add these options for the larger-cluster phase:

```text
--train-threads 4 --codecs cluster-9,cluster-10,cluster-11
```

`phase-provenance.json` preserves the complete original expanded commands,
including original paths and output names. A repeated run still produces new
timings; shared CPU load, filesystem/cache state and FAISS fitting can vary.

## Encoded-byte controls and analysis

The recorded base encoded-byte controls used `cdd6c682`; rate and cluster
controls used `e6d28701`. Each command sets `SOURCE_ROOT` to the corresponding
phase output and a fresh `DEDUP_OUTPUT`:

```bash
"$PY" -m tools.benchmark_faiss_code_dedup \
  --roots "$SOURCE_ROOT" --output-dir "$DEDUP_OUTPUT"
```

Their reports were concatenated, preserving phase commits, unique
`(order, seed, codec)` keys and every all-original-ID verification. Test-score
files were linked into each matching `order-N-seed-S` directory for paired
bootstrap analysis. This merging does not retrain codebooks or alter queries.
The analyzer at `e6d28701` used:

```bash
"$PY" -m tools.analyze_faiss_cotraining \
  --report "$BASE_ROOT/report.json" \
  --supplement-report "$RATE_ROOT/report.json" \
  --cluster-report "$CLUSTER_ROOT/report.json" \
  --dedup-report "$MERGED_DEDUP_ROOT/report.json" \
  --raw-root "$BASE_ROOT" --supplement-root "$RATE_ROOT" \
  --cluster-root "$CLUSTER_ROOT" --dedup-root "$MERGED_DEDUP_ROOT" \
  --output-dir "$ANALYSIS_OUTPUT" --plots
```

For a simpler **new current-checkout measurement**, use
[reproduce.sh](reproduce.sh). It runs the three model phases and one combined
`--roots base rates clusters` encoded-byte control, then the merged analyzer.
It does not reproduce the original phase chronology or old scalar/native-plan
implementation. The current code and new results must receive their own source
SHA and evidence record.

## Paired native-plan optimization

Extract the old source module without changing the current branch:

```bash
git show a068a714d6caf9498ac75a798ff034b8a6a9a815:tools/cotraining_groups.py \
  > "$OLD_CERTIFICATE_MODULE"
```

From a checkout at `106ee3a5`, run:

```bash
"$PY" -m tools.benchmark_bag_certificates \
  --checkpoint-root "$CHECKPOINT_ROOT" --corpus "$CORPUS" \
  --reference-module "$OLD_CERTIFICATE_MODULE" --output "$BATCH_OUTPUT"
```

The recorded default is 15 interleaved repetitions with two warmups per
function, one CPU thread, and source-file hashes checked in the published
batch report. Plan-array equality is verified before timing.

## Inspect published evidence without the original weights

The merged report and analysis contain scalar metrics, hashes, timing samples,
validation selections and test intervals. They contain no weights or raw text.
Decode the compressed JSON with Python's standard library. The four figures can
be redrawn directly from the published numeric analysis:

```bash
"$PY" - <<'PY'
import json
from pathlib import Path
from tools.analyze_faiss_cotraining import figures
root = Path('docs/faiss_cotraining_evidence_20261101')
output = Path('artifacts/faiss-redrawn')
output.mkdir(parents=True, exist_ok=False)
figures(json.loads((root / 'analysis.json').read_text()), output)
PY
```

The displayed figures were redrawn after moving their legend outside the axes
and removing crowded labels; their numeric source remains the frozen merged
analysis. Renderer SHA256 and published file hashes are in `manifest.json`.
Weights are still required to independently remeasure quality and latency or
regenerate paired test intervals from per-pair scores.
