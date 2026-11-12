# Roadmap

Updated 2026-11-12, after the controlled deployment grids. Each item names the
evidence it starts from and the measurement that would close it.

## Where things stand

| Area | Status | Evidence |
| --- | --- | --- |
| Correctness | Original IDs and row bytes preserved across FastText, TinyEngram/Qwen, AG News, DBpedia and the official Engram demo; Linux/macOS wheels | [evidence §3, §5](../evidence/README.md) |
| Capacity | Lossless saving ranges from 0.46% (FastText bundle) and 5.4% (Qwen, v3 format) to about 84% (zero-initialized classifier tables, dedup + Zstd). FP16 native saves 52% vs FP32 | [controlled grids](../evidence/controlled_deployment_results_20261112.md) |
| Query cost | The weak point. With an 8 MiB cache and libdeflate, CPU throughput on canonical Qwen/SGNS traces reaches ~0.92× direct access (0.36× without cache). On classifier tables, host p50 is 29–79× slower than direct FP32 | [CPU results](../evidence/controlled_cpu_results_20261111.json) |
| Structure | Prefix/suffix merging saves 1.34% → 41.49% as training goes from 2k to 32k steps; on a 1.85M-row table extra savings stay below 0.63% | [evidence §2](../evidence/README.md) |
| Identity | Distinct rows beat matched hashing at 20,877 edits (0.41/1.24 pp); no stable gain on smaller grids or held-out PPL | [evidence §1](../evidence/README.md) |
| Construction | Disk-backed builds run in ~100 MiB RSS at 4–5× build time | [report](../evidence/external_incidence_evidence_20261102/README.md) |

## Priorities

### 1. Close the query gap where it is largest

Storage is no longer the bottleneck; lookup speed is. The classifier tables lose
29–79× at p50 while the cached Qwen traces are near parity, so the first step is to
find out why.

- Profile one AG News and one DBpedia query path end to end: Python dispatch,
  ID sorting, block decode, copy. Record where the time goes.
- Test the obvious candidates against that profile: dedup without block
  compression (most of the 84% saving on zero-initialized tables comes from
  duplicate rows, which need no decoding), smaller blocks, row cache, and batching
  at the classifier level.
- **Done when** the selected lossless layout on the controlled AG News/DBpedia CPU
  traces is within 2× of direct FP32 p50 at batch 32, or the profile shows why it
  cannot be and the advisor (item 2) falls back to raw.

### 2. Pick the layout automatically

The grids show the best layout depends on the table: dedup wins on
zero-initialized tables, plain Zstd barely helps on uniform ones, INT8 native
expands, raw is fastest. Users should not have to run a grid to find this out.

- `ngram-store plan --input vectors.f32 --dim D [--trace ids.u64] [--max-slowdown X]`
  estimates dedup ratio and block compressibility from samples, optionally replays
  a trace, and recommends raw, dedup, or a codec and block size.
- **Done when** on the six controlled source models its choice matches the
  calibration-selected layout's capacity within 5% and never violates the
  declared slowdown on held-out traces.

### 3. Packaging

- The wheel currently packages all of `tools/` (185 modules) as
  `meft_memory._runtime`; only the 22 runtime modules listed in
  [`tools/README.md`](../tools/README.md) are needed. Move those into
  `ngram_store/_runtime/` and leave `tools/<name>.py` re-exports, after checking
  that no verifier hashes the current `tools/` files. Experiment scripts stay
  where the evidence expects them.
- One distribution name (`ngram-store`) with `meft-memory` as an alias; publish to
  PyPI.
- **Done when** `pip install ngram-store` works on Linux and macOS and the
  existing CI, including all evidence verifiers, passes unchanged.

### 4. Structure on a long-trained real table

Structural savings grow with training length, but every test so far used
short SGNS runs or a table where they stay below 1%. Before more planner work,
measure one real, fully trained character n-gram model.

- Train or obtain a FastText-scale model trained to convergence on a public corpus
  (e.g. full Wikipedia in one language), with registered keys.
- **Done when** `exact` and `joint` savings and held-out quality are measured on it.
  If `joint` at .99 gives under 10% extra saving at unchanged quality, structural
  compression stays an optional mode and stops being a priority.

### 5. When does identity pay off?

The identity result appears at 20,877 edits and not at 1,024. The next experiment
should test whether the gap grows with capacity pressure (more edits per row
budget) rather than add new tasks. Write the protocol first, as before.

### 6. Technical report

After items 1–2, write up the engine and the three research questions
(identity, structure, storage cost) from the current evidence.

## How changes are measured

Same rows and same query trace against raw mmap and dense lookup. Report total
artifact bytes, p50/p95 latency, load time, RSS and task quality separately.
Freeze the protocol before the run, keep failed and slow runs, and add a
`verify.py` when the result is published. See the [evidence archive](../evidence/README.md)
for the format.

## Using NGramStore in a training pipeline

For a new model with a stable vocabulary: fix unique keys and row IDs during
preprocessing, train a normal mutable embedding table, then freeze it and pack it
with the same IDs. The native backend is read-only; changing the vocabulary means
building a new artifact. Converting an existing hash-bucket model to distinct rows
is a retraining step: bucket rows that were shared during training stay shared.
