# Co-training contribution claims

Updated 2026-11-02. This is a contribution framework for the next paper, not a
claim that the research area is unoccupied. Each statement has its own evidence
and scope. Publication dates document disclosure; they do not establish novelty
over all earlier work.

The strongest paper structure is three linked method contributions:
**characterize usable co-training redundancy, preserve the demonstrated training
updates when compacting it, and certify the inference domain that can consume
the compact parameters.** Storage/backend experiments support those contributions.

## Primary method and empirical claims

| ID | Paper-ready statement | Evidence and scope | Prior foundation / novelty status |
| --- | --- | --- | --- |
| C1 | We characterize an incidence-defined redundancy in learned n-gram tables, separating shared updates from initialization residuals and final-vector similarity. | Complete `(bag, multiplicity)` signatures; matched CPU SGNS on TinyShakespeare and a fixed enwik8 prefix, two gradient rules and three orders/seeds. The new 18-pair replication retains almost-collinear updates, but final cosine is about 0.998 under legacy and 0.796–0.849 under average derivatives. This is not a statement about all prefix/suffix pairs, converged models or all optimizers. | The gradient identity follows from the chain rule. The application-specific measurement and exploitation are the contribution to evaluate; no first-of-kind assertion. |
| C2 | We introduce an incidence-aware compact parameterization with a group-size gradient correction that preserves the demonstrated per-original-row SGD updates and group-mean trajectory in real arithmetic. | FP64 differential tests; paired FP32 training; maximum absolute held-out known-word SGNS relative difference 2.06e-8 on the original study and 5.79e-8 on the independent-corpus replication. Individual initialization residuals are removed. Registered coefficients and optimizer settings must match; row-specific effects and independently gated fields are outside the proof. | Parameter tying and reparameterization are established. The method combines the n-gram grouping criterion with an explicit optimizer/query contract. |
| C3 | We characterize the maximal weight-independent bag-query domain of group-mean projection and provide executable certificates and sparse rejection witnesses. | Equal per-member counts are sufficient and necessary for all finite source tables in real arithmetic. `BagPartition.explain_query` returns a two-row algebraic witness for a rejected bag. OOV rejection is measured separately. A rejected query may still agree for particular weights. | A structured application of nullspace elimination and semantic equivalence, not a new fundamental theorem. The explicit domain/API/counterexamples are the method contribution to assess. |

Sources: [trained evidence](cotraining_evidence_20261031/README.md),
[independent-corpus replication](cross_corpus_evidence_20261102/README.md),
[API and assumptions](cotraining_groups.md),
[derived numerical ledger](cotraining_claims_20261101/claim-ledger.json).

## Supporting claims with narrower scope

| ID | Statement supported by this project | Boundary |
| --- | --- | --- |
| C4 | Certified bags can be compiled into weighted representative schedules, reducing the number of occurrences passed to the native reducer. | The large 512-bag trace represents 21,747 original occurrences with 15,892 weighted group occurrences and a 198,904-byte plan. Fused counting plus cached digest/representatives now improves that identical native total path 3.04×. This is a combined implementation gain, not an isolated incidence operation-count or DATrie effect; the earlier 33.5× codec bypass is separate. Direct known-word precomputation remains the stronger serving control. |
| C5 | The engine offers separate original-row byte-preserving and model-side aggregate-preserving contracts while retaining logical IDs and historical integrations. | Ordinary native storage preserves source bytes. Group-mean transformation changes source rows and admits only certified aggregates. Frozen native handles remain distinct from the mutable PyTorch model. Integration is a systems feature, not evidence of a new model architecture. |
| C6 | Actual large trained tables contain substantial incidence-compaction opportunity, with measured costs that include maps, keys, certificates and query plans. | 1,855,907 updated n-gram rows yield 360,088 groups, or 80.60% fewer parameter-row slots. The large grouped model was not retrained, full word/output weights are unavailable, and construction peaks near 5.4 GiB RSS. This is not a whole-model memory or LLM-quality result. |
| C7 | Production DATrie mark-compact has an independently measured contribution after canonical projection, distinct from global byte dedup and from source-row lossless layout. | On nine canonical small tables DATrie alone adds 11.1–11.8% complete saving; byte dedup alone adds 22.6–36.6%. Gains overlap. Original-byte incidence reorder adds only 0.06674% on the large source. The isolation study does not measure DATrie key-index capacity/latency. |
| C8 | Raw native blocks expose a measured capacity/query-cost tradeoff for certified serving. | Same warm/shared-CPU trace: 529.9 to 15.82 ms, 33.5× faster, with 5.62% more complete bytes than encoded certified storage. Raw mmap and raw precomputed known words remain faster. This is a codec-path result, not a universal native or structural-algorithm speedup. |

These five supporting claims belong in the implementation/evaluation sections;
they should not each be presented as an independent fundamental innovation.

The [FAISS comparison](faiss_cotraining_evidence_20261101/README.md) now tests
513 matched base candidates plus 171 encoded-byte dedup audits on nine actual
trained checkpoints. Within a 0.1% SGNS loss-change, 0.5 percentage-point
discrimination and 5% aggregate-error budget, validation-selected certified
groups save another 11.27% / 33.21% / 34.54% complete bytes over the smallest
qualifying direct codec at orders 4/8/12. All selected test point estimates
pass; the extended grid is exploratory. Same-codec grouped reconstruction is
less accurate, full-domain hybrid costs more than direct, and precomputed known
words are smaller/faster for inference only. This supports complementarity to
quantization within the certified domain, not universal FAISS dominance or a
new primary novelty claim. Batch certification also improves native-plan
construction 3.1–3.4× without changing its arrays; that is an implementation
result distinct from the method claims.

The [frozen-codec query follow-up](faiss_schedule_evidence_20261102/README.md)
also measures compilation plus FAISS bag query, rather than inferring it from
certificate-only speedups. Batching reduces known-word total path time about
3.2–3.4× with byte-identical outputs; direct SQ6 remains faster. This adds
systems evidence for the supported query domain, without changing quality,
capacity or novelty claims.

The [predeclared independent-corpus follow-up](cross_corpus_evidence_20261102/README.md)
adds 18 paired training runs, all 513 base codec candidates and 171 direct-code
dedup audits. Validation-selected groups save **12.33% / 37.12% / 40.47%**
complete bytes at the same primary quality budget. All primary test point
estimates pass, while the tighter 0.01% budget has four failures. Fused counts
are also applied to the direct controls: registered 8/12-order total queries
are now faster than selected direct SQ6 on both corpus prefixes; 4-order is
near parity or slightly slower. Same-codec reconstruction remains less
accurate, and full-domain hybrid remains larger/slower. Precomputed words
still dominate restricted inference. This strengthens cross-corpus method and
systems evidence, without establishing universal accuracy, serving or novelty
dominance. The actual large native table is retimed, not retrained.

## Precise certificate result

Let `P` replace each group's rows with their mean, `E` be arbitrary source rows,
and `c` be the integer count vector of a bag. The sum is preserved for every `E`
exactly when:

```
c^T P = c^T
```

For a mean-projection block, this is exactly equality of all member counts.
If `c_i != c_j` in a group, take a one-dimensional table with `E_i = +1`,
`E_j = -1` and zero elsewhere. Then `P E = 0`, while `c^T E = c_i - c_j != 0`.
This proves necessity as well as sufficiency without assuming high cosine.
The result is uniform over source rows; it does not deny coincidental equality
for a rejected query on a specific checkpoint. FP32 rounding is measured
separately from this real-arithmetic result.

The [diagnostic tests](cotraining_claims_20261101/query-diagnostic-tests.txt)
enumerate 243 finite count combinations and verify every rejection witness;
they also cover a missing member beyond a map-scan chunk boundary. Those are
deterministic algebra/correctness fixtures, not compression or semantic data.

## Prior work that these claims must acknowledge

- [HashedNets (2015)](https://arxiv.org/abs/1504.04788), section 4.4:
  equality-constrained parameter sharing and aggregation of shared gradients.
- [Net2Net (2015/2016)](https://arxiv.org/abs/1511.05641), section 2.3:
  function-preserving replication and weight compensation.
- [Lossless Compression of Deep Neural Networks (2020)](https://arxiv.org/abs/2001.00218),
  algorithm 1 and theorem 3.1: remove redundant units while preserving outputs
  on a declared input domain.
- [Bisimulations for Neural Network Reduction (2021/2022)](https://arxiv.org/abs/2110.03726),
  definitions 6–7: equivalence partitions and semantically equivalent quotients.
- [FastText.zip (2016)](https://arxiv.org/abs/1612.03651), sections 3.2–3.3:
  compressed embeddings, retraining and feature pruning with document coverage.
- [The original me_fasttext preprint (2025)](https://arxiv.org/abs/2506.01254):
  the project's DATrie/prefix-suffix sharing/mark-compact lineage. The current
  incidence/optimizer/certificate method is a later extension.

This bounded review did not identify the same complete n-gram pipeline in these
representative sources. Citation tracing and a wider search are still required
before using even a narrowly qualified “first” statement.

## Research opportunities that are not result claims yet

| Priority | Candidate contribution | Evidence required before claiming the result |
| --- | --- | --- |
| 1 | A construction algorithm that exploits incidence without full in-memory signatures/row copies. | External-memory/chunked implementation; fixed RAM-budget sweeps, peak RSS/temp disk/build time, all-row validation and the current Python builder as a baseline. O(N) on-disk maps do not imply O(N) resident RAM, and chunking alone does not prove bounded construction. |
| 2 | Proportional-incidence groups beyond exactly identical columns. | Detect proportional integer signatures; derive the weighted parameterization and squared-coefficient gradient correction; prove query conditions; run paired SGD and real-table complete-cost controls. This is related to feature factorization and must be checked against prior work. |
| 3 | Certified approximation for partially used groups at an explicit query-error budget. | Derive bounds from actual residuals and query coefficients, include rounding/metadata cost, compare accepted-query coverage and downstream quality at fixed storage. Cosine/norm-ratio thresholds alone do not establish such a bound. |
| 4 | Role-specific incidence compaction for learned LLM memory. | Trace each field's actual coefficients and gating; preserve fields separately; paired trained-model quality/capacity and head-aware controls. Existing raw-row Engram/TinyEngram integration does not validate model-side group sharing through independent gates. |

These are explicit research hypotheses and engineering targets that require
implementation and evaluation. For the current paper, use “we propose / characterize / measure” and
attach assumptions and evidence to each statement.
