# Prefix/suffix node sharing: quality, capacity and complete query protocol

Fixed on 2026-11-09 before these new measurements. Existing conditional
similarity, SGNS and codec results are known. This follow-up is exploratory
with respect to the previously used enwik8 validation/test prefix; an additional
confirmation segment is fixed below before scoring. No result is discarded for
contradicting a desired compression or length trend.

## Sources and task

Use all 18 existing dense SGNS checkpoints: enwik8 maximum orders 4/8/12,
seeds 3/7/11, legacy and standard average derivatives. Verify the published
checkpoint/corpus/source hashes. Preserve word/output weights; replace only
the trained n-gram row storage. There is no new random-vector performance table.

Keep the original 160k/20k/20k token train/validation/test split. From the same
cached, hash-verified enwik8 ZIP and the same first 4 MiB/regex preprocessing,
fix token positions [200000,300000) as a new 100k-token confirmation segment.
It is held out from these models and earlier 200k-prefix scoring, not an
independent corpus or untouched external dataset. Do not fetch a new corpus.

Score ALL adjacent-window context pairs with a registered output target in
each evaluation segment, including OOV input centers. Known centers use the
word row plus registered n-gram multiplicities; OOV centers use their registered
n-grams only. Report known, nonempty-OOV, empty-OOV and all-center strata,
target-vocabulary coverage and stored-ngram coverage. Keep original positions
and OOV gaps. Use five fixed frequency^.5 sampled negatives, including recorded
target collisions, and paired float64 SGNS losses/context discrimination.
This measures a finite subword context model, not classification, semantic
benchmarks, full softmax perplexity, Engram pretraining or LLM quality.

## Complete fixed candidate grid

For every checkpoint compare the same original-ID domain:

- Raw FP32/FP16/symmetric per-row INT8 (scale stored with each row), and a
  dense in-memory FP32 control.
- Native identity layouts in original and DATrie-planned order, all three
  precisions, plus native exact encoded-byte deduplication controls.
- Prefix/suffix tree parent-child and multi-level node sharing, cosine gates
  .99/.98/.97/.95/.90 crossed with norm-ratio gates .99/.98/.97/.95/.90.
  Run both the original C++ DATrie two-pass mark/compact planner with its final
  representative guard and the independent shortest-first, best-retained-node
  reference. Encode every mapping in FP32/FP16/per-row INT8. Gates apply to
  original vectors; report the final error after additional quantization.
- The complete 19-config existing FAISS grid: FP32, FP16, SQ8/6/4, full-vector
  clustering with 256/512/1024/2048 centroids, PQ8x4/8x8/16x8/32x8/64x8,
  OPQ8x8/32x8/64x8, RQ2x8/4x8. Fit only checkpoint rows, using the same
  sampled original IDs (up to 16384), seed 42, 20 codec/10 OPQ iterations.
  Include an exact encoded-byte dedup alternative for EVERY direct codec,
  with the smallest unsigned original-to-code map and the same codebook.

Every candidate retains original addressing and serves every input string;
empty stored-ngram bags are explicitly reported. Reopen each produced bundle
and verify all reconstructed rows and finite values. Keep negative size,
quality and construction results. Byte-identical layouts may share measured
artifacts only after hash/map/codec equality is checked and all requested
candidate names remain in the ledger. This is a finite predeclared grid, not
exhaustive tuning or a proof of universal FAISS superiority.

## Fair frozen-inference controls and costs

Use the same original-ID DATrie index and reducer in the primary lookup
comparison, including the index in serving bytes. Also retime selected bundles
with the independent hash lookup index to isolate indexing cost. Complete model
bytes include the key/word registries, word/output rows, row codes, native
descriptors/maps, quantizer scales, codebooks and metadata. Shared build inputs,
reference arrays, scoring data and original checkpoints are audit assets, not
silently included or excluded from a candidate's serving cost.

Measure both ordinary subword inference and a stronger full-domain hybrid:
precompute original full aggregates for ALL registered words, replacing the
same-size word-row file; OOV inputs still use the candidate row backend.
The hybrid's registered-word aggregates are exact to the common source reducer.
Keep the known-word-only precomputation boundary visible; do not present a
known-only serving advantage as an arbitrary n-gram requirement.

Validation alone selects the smallest complete bytes at loss increases
.01%/.1%/.5%/1%/2% in both all-center and nonempty-OOV strata; .1% is primary.
Require discrimination decline <=.5
percentage point in both all-center and nonempty-OOV strata. Keep the selected
candidate frozen for original test and new confirmation. Record pointwise
paired 128-token-block bootstrap intervals (2000 resamples, fixed seed 42).
No independent-pair p-values or simultaneous/generalization guarantee.

Use deterministic lexical samples of up to 512 known and 512 OOV confirmation
types, fixed seed 42, preserving empty bags. In fresh one-thread processes,
measure string-to-key lookup, original-ID count/schedule construction, selected
row decode and weighted aggregate together, plus precompiled replay separately.
Use 64-input batches, two warmups and 15 randomized repeats, report p50/p95,
per-batch samples, load time and anonymous/file-backed/peak RSS. Native row
blocks use the existing 8192-byte maximum uncompressed block budget, keeping
its precision-specific block-row limits visible. Selected
bundles get a separate matched 0/1MiB native-cache retiming; table bytes and
quality remain unchanged. Warm shared CPU results are not exclusive/disk-cold
or production latency. Record per-candidate fresh build peak RSS, wall time,
artifact files and compiler/library/source hashes.

## Training-mechanism factorial

Using the same training-only vocabulary and source initialization, run both
gradient conventions, all three orders/seeds, n-gram-only initialization
scales 1/.1, and 2000/8000/32000 SGD steps. This is 36 trajectories with
three recorded durations, 108 model points. Word initialization is unchanged;
output initialization remains zero. All conditions share pre-generated context
and negative schedules and their hashes. Reproduce the original 2000-step
scale-1 row/output bytes before extending; if replay fails, preserve the
diagnostic and repair the schedule instead of treating it as a matched run.

At every duration report both endpoint-frequency/length/coverage diagnostics,
accumulated-update versus initialization norms, actual common-update coverage,
joint .99/.99 compaction and complete held-out quality. Fixed checkpoints are
not validation-selected training durations, and no convergence claim follows.
Only validation may select any capacity operating point; retain all seeds,
weak effects and reversals. This distinguishes update exposure and
initialization sensitivity from an unsupported universal length rule.

## Execution and publication

CPU only, empty CUDA_VISIBLE_DEVICES. Write exclusively in a new task directory
under /llm-align/duyimin; original sources are read-only and GPU processes are
untouched. At most nine independent model jobs, up to four fitting threads
each; query workers have one thread. Preserve contention/load metadata and
stage final selected retimings after this task's builders/training exit.

Record complete source SHA, commands, Python/PyTorch/NumPy/FAISS, OS/CPU,
compiler, threads, source/data hashes, raw measurements, failures and exact
cost assumptions. Use existing hardware/assets with no paid service; energy
and monetary metering are not available. Publish aggregate results, paired
score/block evidence, reproducible code snapshots and a stdlib verifier;
do not publish corpora, raw token/string keys, weights or internal hostnames.
Finish the complete requested grid, quality audits, query measurements,
mechanism experiment, figures, evidence verifier and required CI before merging.
