# Frozen trained-checkpoint partial-query protocol, 2026-11-04

Save this protocol and implementation commit before measuring the new path.
This is an exploratory CPU follow-up using existing source-linked trained inputs,
not new training, independent adoption, convergence or downstream quality.

## Inputs and fixed choices

- Reuse the cached enwik8-prefix SGNS checkpoints from
  [the independent-corpus study](cross_corpus_evidence_20261102/README.md).
- Maximum orders 4/8/12, seeds 3/7/11: nine legacy-convention checkpoints are
  primary; the nine average-derivative counterparts are a sensitivity control.
- Verify each checkpoint and corpus SHA256 against the published training report.
  No external data/model downloads, text/weight redistribution or GPU operations.
- Use unchanged original training bag multiplicities, n-gram-only rows and
  training-only vocabulary. Each query accesses only already-stored n-grams;
  missing n-gram types are omitted identically in all references.
- Known/OOV traces follow the earlier seed-42 selection: contiguous held-out
  test split, sorted unique words, up to 512 sampled words per trace. Preserve
  the exact trace digest and report empty bags separately. Exclude empty bags
  from coverage denominators, never count rejection as a completed query.
- Fixed relative-L2 budgets: 0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05,
  0.1, 0.2, 0.5, 1.0. Do not choose/revise a budget from measured test results.
  These are per-query aggregate-L2 budgets, not cosine or SGNS-loss budgets.

## Measurements

1. Build the existing raw native certified store from FP32/CSR input, then its
   separate actual-source residual sidecar. Record full output hashes, sizes,
   build wall time/RSS and logical scratch. Final native row bytes must match
   the legacy projection; the sidecar must not modify the original store.
2. Compare strict membership coverage to source-specific budget coverage.
   For every candidate, measure returned sum/mean errors against FP64 source
   aggregation, and check absolute/relative errors against the computed bounds.
   Floating oracle checks supplement exact-fraction arithmetic unit fixtures.
   Any violated bound is a failure, not a candidate to omit or a threshold to tune.
3. Report coverage, newly admitted queries and maximum observed error among
   admitted nonempty queries for every predeclared budget. Report conservative
   bound rejection separately from actual-error oracle eligibility.
4. Count complete native table, certificate, manifest and sidecar bytes. Include
   unchanged original word/output FP32 rows and UTF-8 registry bytes in the
   complete-model comparison to source raw FP32. Query-only metadata and plans
   are separate. No source fallback is retained; rejected queries remain rejected.
5. Record native strict accepted-query cost and new full assess cost on all trace
   queries, including rejected candidates, plus source FP64 reference cost. Use
   two warmups, five repetitions, one CPU thread, and state different scopes.
   Bounds/candidates are assessed once then budget selection is reported
   separately; this is not an exclusive service or cold-disk latency claim.
6. Publish per-checkpoint metrics, all selections/failures, commands, source/input
   digests, Python/NumPy/Torch versions, OS/CPU and cost limitations. No money or
   energy measurements. Source inputs and temporary source/stores stay remote.

## Interpretation

Evidence can establish actual-source partial-query coverage at measured capacity
and query costs on these finite checkpoints. It does not extend the weight-
independent equality theorem, establish arbitrary OOV/model accuracy, or prove
all same-prefix/suffix final weights are 99% similar. Report both conventions
even when one is materially weaker. The additional cost of checking bounds is
part of the result, including regressions against strict/native/dense controls.

## Frozen performance follow-up

The first complete run validates the bounds but exposes per-bag Python norm
loops as a bottleneck. Before retiming, freeze its full report and source. Batch
native row gathering and interval norms, use scalar outward rounding for scalar
bounds, and skip groups with zero residual mass. Keep all query arithmetic order,
source/store/sidecar bytes and budgets. Compare output and every bound/domain
array byte-for-byte on all 18 checkpoints. Time the nine primary legacy
checkpoints only, both known/OOV traces, two warmups and five paired repetitions,
seed-42 randomized reference-first/batched-first ordering, one CPU thread.
Sensitivity checkpoints perform equality checks only. Preserve the first slow
results, publish both implementations' exact source/report hashes and retain
the dense/strict-subset comparisons with their different query scopes.
