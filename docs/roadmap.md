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
- **Progress (2026-11-12).** The [local profile](../evidence/host_query_profile_20261112/README.md)
  found the time is block decompression: 256-row Zstd blocks and scattered IDs.
  On the zero-init AG News table, dedup without compression saves 85.8% (86.6%
  with Zstd) at 1.2× raw mmap (was 356×); on the uniform-init table, raw is the
  right layout. `NativeRows.gather_array` removes the Python overhead that then
  dominates, and `pack --no-compress` exposes the layout. On the twelve original
  AG News/DBpedia tables the full GPU host path runs 1.09–1.23× raw with identical
  logits, so the exit criterion is met. Item 2 (choosing the layout) is next.

### 2. Pick the layout automatically

Done (2026-11-12). `ngram-store plan` counts duplicate rows, gives exact sizes
for raw and deduplicated uncompressed layouts, and optionally estimates compressed
ones. On the twelve controlled tables its uncompressed sizes are exact, its
compressed estimates are within 0.15%, and `--prefer size` reproduces the grid's
calibration choice of codec family on all twelve
([validation](../evidence/host_query_profile_20261112/README.md#layout-planner)).
Not done: replaying a query trace to predict latency for compressed layouts.

### 3. Packaging

Done in source (0.5.0): the distribution is `ngram-store`, and wheels ship only
the 22 runtime modules, computed at build time as the import closure of the
public entry points (`setup.py`). `tools/` is unchanged on disk, so evidence
verifiers and `tools.*` imports keep working. Not done: a PyPI upload and a
`meft-memory` alias package, which need a release decision.

### 4. Structure on a long-trained table

Cheap version done (2026-11-13): on the existing 32k-step checkpoints, joint
sharing stored as dedup + uncompressed keeps 42.4% of the 46.0% n-gram saving with
lookups at ~2× raw instead of ~125×, and no measurable loss change
([report](../evidence/structure_dedup_20261113/README.md)). A converged
FastText-scale table and a FAISS comparison at these points remain open.

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
