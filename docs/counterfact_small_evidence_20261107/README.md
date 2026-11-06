# MultiCounterFact: completed small grid and native fidelity

The complete 1,024-edit three-seed grid **does not establish a need for the large
memory table**. Large-versus-small/mid/head-constant/memory-off harmonic-score
intervals include zero for both models. The parameter-matched adapter wins on
0.6B; the 4B comparison is inconclusive. Calibration-selected BM25 retrieval has
the highest observed harmonic score for both models. Native exports preserve
bytes but slightly expand on disk.

These are official MEMIT MultiCounterFact **counterfactual editing** results,
not verified contemporary facts, editing-method SOTA, language-model pretraining,
or proof of a storage advantage. The 20,877-edit grid was still generating prefix
caches at this snapshot and is excluded. Negative results and independently
specified exploratory runs are retained.

## Protocol and coverage

- Main: Qwen3-0.6B and Qwen3.5-4B; first 1,024 official edits; seeds 3/7/11;
  six independently trained branches/controls per model. Train the last eight
  original decoder layers and branch for ten fixed epochs with masked rewrite
  new-target/EOS loss. No paraphrase or neighborhood training.
- Completed: 36 trained runs, two unedited bases, two BM25 controls. Each scores
  all 1,024 cases. Test: 921 cases, 1,842 paraphrases, 9,210 neighborhoods.
  Both folds' rewrite facts are edited, as in knowledge editing; this is not
  held-out-fact generalization.
- Retrieval: inverted-index BM25 of edited rewrite/new facts only; top-1/top-3
  selected by calibration harmonic score, smaller-k tie-break. Both models
  selected top-1. Evaluation prompts and answers are not indexed.
- Exploration: whole backbone frozen, large/adapter, seed 3, twenty epochs and
  separately registered learning rates. Four runs; this single seed does not
  replace the main experiment or establish robust superiority.
- Interface: sixteen actual official question prefixes per model, without
  answer tokens, plus every exported row. Independent reopen uses the original
  base, nonrow delta and native rows. This is a limited next-token fidelity check.

Exact commands and environments are in reports. Main commit `537debf`, probe
commit `d6d828a`, export/analysis commit `b81bfaa` are archived under [runtime](runtime/).
The [main protocol](../../benchmarks/counterfact_capacity_v1.json) SHA256 is
`1a33c174723ada0a991130de77cdc06177da6de003319049fd9daa2732106825`;
the [exploratory protocol](../../benchmarks/counterfact_memory_only_probe_v1.json)
SHA256 is `62b3ee9f6489800bde73fbca370f934e626d044474c30836b2f50c39c9b2ad35`.
Data SHA256 is `977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833`.

Hardware: A100-SXM4 80GB, AMD EPYC 7742, four PyTorch CPU threads and one interop
thread per job; Python 3.11.15, Torch 2.11.0+cu130, Transformers 5.3.0, NumPy
2.2.6. FP32 parameters/optimizer state and disabled PyTorch TF32 flags are
recorded; the installed FLA triangular solver explicitly uses TF32. The earlier
[precision audit](../counterfact_capacity_evidence_20261106/precision-audit.json)
applies. Concurrent jobs and prefix-cache reuse make run times unsuitable as
exclusive-GPU serving benchmarks.

## Main quality results

Percentages. Trained ES/PS/NS average official per-case macro scores over three
seeds; H is the harmonic of those three averages, not mean per-seed H. Base and
BM25 are one fixed evaluation each. Full-target teacher-forced accuracy and
both calibration candidates remain in the reports.

| Model | Variant | ES | PS | NS | H |
|---|---|---:|---:|---:|---:|
| 0.6B | Unedited base | 26.60 | 28.77 | 71.86 | 34.78 |
| 0.6B | Small | 100.00 | 72.11 | 43.49 | 64.02 |
| 0.6B | Mid | 100.00 | 71.95 | 43.68 | 64.11 |
| 0.6B | Large | 100.00 | 72.19 | 43.63 | 64.14 |
| 0.6B | Head constants | 100.00 | 72.49 | 43.47 | 64.11 |
| 0.6B | Memory off | 100.00 | 72.64 | 43.17 | 63.92 |
| 0.6B | Matched adapter | 100.00 | 78.25 | 46.20 | 67.53 |
| 0.6B | BM25 top-1 | 100.00 | 88.60 | 65.58 | 82.11 |
| 4B | Unedited base | 16.07 | 17.26 | 83.59 | 22.71 |
| 4B | Small | 100.00 | 87.50 | 45.23 | 68.90 |
| 4B | Mid | 100.00 | 87.71 | 44.98 | 68.76 |
| 4B | Large | 100.00 | 87.71 | 44.84 | 68.65 |
| 4B | Head constants | 99.96 | 87.84 | 45.34 | 69.05 |
| 4B | Memory off | 100.00 | 87.31 | 44.62 | 68.39 |
| 4B | Matched adapter | 99.93 | 90.75 | 43.43 | 68.10 |
| 4B | BM25 top-1 | 99.89 | 86.70 | 71.92 | 84.63 |

Main training improves rewrite efficacy while damaging locality. High rewrite
scores alone do not establish a useful memory mechanism. BM25 adds context
tokens and an index; its observed quality win is not a latency/cost Pareto claim.

Paired large-minus-control H, percentage points. Resample training seeds and
the same case IDs for both sides; 2,000 replicates, fixed RNG. Three seeds are
limited, and shared subjects/neighborhoods make case independence an assumption.
These intervals have no multiple-comparison correction.

| Control | 0.6B delta [95% interval] | 4B delta [95% interval] |
|---|---:|---:|
| Small | +0.12 [-0.33, +0.61] | -0.26 [-1.58, +0.87] |
| Mid | +0.03 [-0.30, +0.39] | -0.11 [-0.58, +0.38] |
| Head constants | +0.03 [-0.40, +0.52] | -0.41 [-1.39, +0.51] |
| Memory off | +0.21 [-0.43, +0.95] | +0.26 [-0.47, +0.93] |
| Matched adapter | -3.40 [-4.38, -2.38] | +0.55 [-0.61, +1.78] |

Small/mid/large have 3,628 / 57,384 / 917,516 physical FP32 rows of dimension
32. Large rows alone cost 117,442,048 bytes. The main delta also includes the
decoder suffix: large costs 593.61 MiB on 0.6B and 3,520.51 MiB on 4B.
The adapter is parameter matched within 384 / 2,432 parameters respectively.
Main deployment cost cannot be represented by row bytes alone.

## Separate frozen-backbone exploration

One seed and a different schedule, with no cross-protocol checkpoint selection.
Suffix hashes and masked-gradient checks confirm the whole backbone stayed fixed.

| Model | Variant | ES | PS | NS | H |
|---|---|---:|---:|---:|---:|
| 0.6B | Large | 100.00 | 51.30 | 60.30 | 65.11 |
| 0.6B | Matched adapter | 99.89 | 79.32 | 43.09 | 65.47 |
| 4B | Large | 100.00 | 52.88 | 58.49 | 65.21 |
| 4B | Matched adapter | 98.15 | 88.33 | 36.09 | 60.95 |

Freezing changes the efficacy/generalization/locality tradeoff. The 4B large
branch's higher H is a one-seed observation, still below the original base on
locality and far below main BM25 on H. It does not establish large-versus-small
capacity necessity.

## Native export: fidelity passes, storage expands

Native XOR, block size 32, deduplication disabled, original hash IDs retained.
The row map costs 7,340,128 bytes. Both exports verified all 917,516 rows byte
for byte and sixteen source/native next-token outputs byte for byte. Independent
native-bundle reopen matched every checked request.

| Model | Raw row bytes | Native table bytes | Source delta bytes | Native bundle bytes |
|---|---:|---:|---:|---:|
| 0.6B frozen probe | 117,442,048 | 118,542,991 | 119,048,616 | 120,153,490 |
| 4B frozen probe | 117,442,048 | 118,484,784 | 121,457,072 | 122,503,741 |

Bundle bytes include the table, metadata, nonrow weights and model recipe;
the validation report is excluded. These trained hash tables have not
demonstrated a compressible row structure. This checks the generic original-ID
row backend, **not** a DATrie/mark-compact string-key compression gain or the
joint-training prefix/suffix similarity hypothesis. Existing hash collisions
are preserved rather than removed.

Two pinned staging slots total at most 1 MiB; the native row cache is 1 MiB.
Validation keeps source/native models together, so no comparative peak-memory,
throughput, latency, generation or KV-cache claim follows. Reopen constructs
a temporary dense row module first; bounded startup is not established.

## Inspect and reproduce

```bash
python -S docs/counterfact_small_evidence_20261107/verify.py
# Optional: NumPy reproduces all twelve paired bootstrap intervals exactly.
python docs/counterfact_small_evidence_20261107/verify.py --bootstrap
```

The [manifest](manifest.json) pins raw reports, gzip case scores, logs, commands
and runtime snapshots. Decompression yields original report hashes. Both
candidate NLLs and selected-target correctness remain per prompt; duplicate
sequence-index dictionaries are omitted. Weights/deltas, token IDs and prefix
states remain private. The verifier checks success against NLLs, recomputes
all scores and paired point estimates, checks both RAG candidates and calibration
selection, and pairs native exports to their source deltas. It does not rerun
CUDA or inspect withheld weights.

[Small analysis](small-analysis.json), [probe analysis](probe-analysis.json),
[queue snapshot](queue-snapshot.json), [CPU/dependencies](hardware.json) and
[native build](native-build.log) provide provenance. Near ties at 0.0014 NLL and
the diagnostic excluding neighbors that are also edited rewrites remain in
analysis; official metrics retain every prompt. No main training/evaluation
failure occurred in this snapshot. Earlier fixture/preflight failures remain
described in the [preflight archive](../counterfact_capacity_evidence_20261106/README.md).

Implementation and results: [PR #54](https://github.com/initial-d/ngram-store/pull/54).
The next decision depends on the frozen full-edit grid, not test-based selection
of a new architecture or codec.

## Subsequent cache logging failure and recovery

After the small-grid snapshot, all six full 0.6B prefix caches saved complete
manifests and data, then their final JSON log print failed on a `set` returned
in Transformers loading information. The finite queue correctly stopped and
its helper exited. This was a failure, even though cache artifacts had finished;
the [original logs and queue outcomes](recovery/failure-1/) are preserved.

The fix prints the same normalized JSON representation as the saved manifest.
An explicit resume validates hashes, tensor sizes, contiguous offsets, exact
per-shard prepared sequence coverage and the immutable-prefix hash before
reusing these caches. Completed small-grid weights and scores are also checked.
No learning rate, edit set, training schedule or scoring rule changes. Recovered
artifacts are labeled reused; original nonzero process exits remain in history.
