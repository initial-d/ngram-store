# Same-task FAISS / incidence-compaction protocol

This compares original-ID n-gram gathering and bag aggregation, **not nearest
neighbor search**. FAISS supplies standalone codecs: selected per-row codes are
decoded without expanding the whole table. IVF/HNSW search latency and recall
are outside this task. Results belong to a finite SGNS reference, not learned
Engram quality or semantic/PPL evidence.

## Sources and fitting

Reuse the nine existing TinyShakespeare SGNS checkpoints: maximum orders 4/8/12
and seeds 3/7/11, 128 dimensions, 1,200 training words, 2,000 × 256 sampled
training pairs. Vocabulary/counts come from the original contiguous 80% token
training split. Validation/test use the next two 10% splits, retaining OOV gaps
and original positions. These splits were used in earlier project studies;
they are held out from original model training, not a fresh external validation.

Only actual checkpoint rows are encoded; no synthetic vectors, row duplication
to inflate scale, retraining or corpus downloads. A finite trained checkpoint
can retain initialization in some parameters; convergence is not asserted.
Word rows and output embeddings remain FP32 and unchanged in every n-gram codec
control. Padding is excluded. Complete `(bag ID, multiplicity)` signatures
define groups; merely sharing a prefix/suffix is insufficient.

Train each codebook on at most 16,384 original n-gram IDs sampled without
replacement using seed 42. A grouped codebook sees the corresponding group means
at those SAME sampled IDs, preserving original-row occupancy weighting instead
of uniformly sampling representatives. Fitting reads no validation/test bags,
outputs, losses or labels. Codes for all stored rows are written in chunks.
Serialized FAISS indexes have `ntotal=0`; codebooks/transforms and row codes are
counted separately, without silently storing a duplicate full index.

## Rate grid and disclosure

- Raw FP32; FAISS FP16 and 8/6/4-bit scalar quantizers.
- Full-vector 256/512/1,024/2,048-centroid clustering (one PQ subquantizer).
- PQ: 8 × 4, 8 × 8, 16 × 8, 32 × 8 and 64 × 8 bits.
- OPQ + PQ: 8 × 8, 32 × 8 and 64 × 8 bits.
- Residual quantization: 2 × 8 and 4 × 8 bits.

K-means uses 20 iterations, seed 42; OPQ uses 10 outer iterations and 20 PQ
iterations. FAISS's residual beam settings remain default and its version is
recorded. Small-table centroid-training warnings are preserved. This is a fixed
rate sweep, not exhaustive per-codec tuning. The base grid fits with one CPU
thread; the seven-codec rate extension fits with four after observing expensive
single-thread OPQ construction. All query workers use one thread. Do not compare
their fitting times as a matched-thread speedup. The three larger clustering
configurations also fit with four threads, as a separately disclosed extension.

The initial nine-codec protocol preceded its full run. Seven rate extensions
were added after initial aggregate errors showed that the original PQ settings
did not reach the low-error region. First-phase test summaries were already
available. The combined rate sweep and validation-selected test audits are
therefore exploratory, not a prospectively untouched final test claim. Each
phase has its own frozen source, command and protocol; matching checkpoints,
incidence groups, sampled fit IDs, evaluation hashes and query traces are
checked before merging. Test scores do not enter the selection function.

Completed results, point-estimate and uncertainty limitations, stronger
baselines and source-pinned commands are in the
[2026-11-01 evidence report](faiss_cotraining_evidence_20261101/README.md).

## Query contracts and complete cost

For every codec compare:

1. **Direct**: all original n-gram IDs are supported, with codec approximation.
2. **Certified group**: group means are encoded; incomplete/unequal group
   compositions are rejected. FP32 groups preserve certified sums in real
   arithmetic, with observed FP32 rounding.
3. **Full-domain hybrid**: rejected bags use original FP32 rows for every
   nonsingleton member; singleton rows still use the selected codec. Store ALL
   nonsingleton original rows, not just those touched by the measured trace.

Complete model cost includes word/output rows, the same word+n-gram JSON
registry, codes, FAISS codebooks/OPQ transforms, codec metadata, group domains
and the full hybrid fallback. Fixed-trace plan files are reported separately;
audit checkpoints, query input lists, scoring arrays and build logs are not
serving bytes. Temporary-disk peak and a bounded-memory builder are not claimed.

All controls deduplicate repeated IDs and retain multiplicity. Certified groups
additionally reduce equal group occurrences to weighted representatives.
All use the same FP64 weighted reducer and FP32 output. Compare a deterministic
sample of up to 512 held-out known word types and 512 OOV types; distinguish
empty OOV bags from nonempty certified coverage. Rejected queries never count
as successful accelerated work.

Report two timing modes: precompiled schedule replay, and schedule compilation
from original IDs plus query. Each candidate uses a fresh process, one CPU
thread, two warmups and 15 replays. The shared host and warm OS pages do not
establish exclusive/disk-cold performance; the two rate phases may overlap on
the shared machine. Record p50/p95, samples, acceptance and RSS before full
quality decoding. Cross-phase timing frontiers are descriptive, not controlled
exclusive-speed comparisons. No GPU process is inspected or modified.

The [2026-11-02 frozen-codec follow-up](faiss_schedule_evidence_20261102/README.md)
retimes old scalar and new batch certification on prior selected artifacts.
Its compilation-plus-query results are separate from the frozen timing grid;
no codebook or quality selection is rerun.

## Quality and stronger controls

Measure all registered known-word aggregate relative L2, query error and
held-out SGNS loss/context discrimination against the same original checkpoint.
Use 8,192 sampled pairs per split and five fixed negatives; both candidates and
reference use the same FP64 objective implementation. These SGNS scores do not
measure word analogy, retrieval quality or LLM loss.

Select the smallest complete model **on validation**, at loss increases of
0.01%, 0.1% and 0.5%, a maximum 0.5 percentage-point discrimination drop, and
at most 5% known-word aggregate relative L2. Audit the selected candidates on
test. Paired 128-token-block bootstrap intervals are within-text correlation
approximations, not independent-corpus or simultaneous guarantees.

Two separate serving controls prevent weak-baseline conclusions:

- Precompute every known word's original full aggregate once, retain outputs
  and a word-only registry, and gather those vectors directly in a fresh process.
  This is a strong known-word inference baseline, with no OOV, arbitrary n-gram
  lookup or matched training contract.
- Serialize the group map using the smallest unsigned integer width and verify
  the complete partition summary after loading. Report actual storage as a
  separate ablation. Timing measurements use the original uint64 domain; the
  in-memory map remains int64, so narrower on-disk maps do not establish a
  query-memory or latency gain.

A further full-domain control deduplicates identical **encoded row bytes** after
direct FAISS quantization, using the smallest unsigned original-to-code map.
It reuses the same codebook, verifies every original-ID reconstruction and
collapses repeated code IDs in all bags, without incidence restrictions or an
FP32 fallback. Only candidates smaller than the matched complete direct bundle
receive fresh query/quality workers; negative size results select the original
flat codec and remain in the audit. This is exact relative to the codec's bytes,
with separately checked OPQ batch-rounding differences, not lossless to FP32.
Beneficial candidates participate in the direct baseline's validation selection.
This control preserves frozen inference, not the original training parameters.

## Run

Install optional dependencies with `pip install '.[faiss-benchmark]'`. With the
cached corpus/checkpoints, run on a CPU-only environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.benchmark_faiss_cotraining \
  --checkpoint-root "$CHECKPOINT_ROOT" --corpus "$CORPUS" \
  --output-dir "$OUTPUT_ROOT" --orders 4,8,12 --seeds 3,7,11
```

Use the evidence report's exact phase-specific commands to reproduce the
recorded runs. The expanded current default combines both rate grids in a new
single run; that is a new measurement, not an identical replay of two phases.
