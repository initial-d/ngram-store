# Native bounded-plan follow-up, 2026-11-05

Freeze this protocol and implementation before measuring. Reuse the 18 passed
enwik8-prefix finite SGNS sources, native bundles, residual sidecars and exact
known/OOV traces from the 2026-11-04 partial-query study. No new text/model
downloads, training, GPU inspection or GPU process changes. Work only in a task
subdirectory under `/llm-align/duyimin/`, using the provided Python environment.

Compare `BudgetedBagRows(compiler="python")` with `compiler="native"` on the
same compiled C library, unchanged table/sidecar files and input bags. The C
addition performs residual-bound centering, exact-domain checks and representative
plan construction; counting input IDs, native reading/reduction and remaining
norm/rounding calculations keep their original contracts. `auto` uses C when
present and falls back for older libraries.

- All legacy/average checkpoints, orders 4/8/12 and seeds 3/7/11: compare sum and
  mean candidate vector, projection/rounding/absolute/relative-bound and exact-
  domain arrays byte-for-byte. Compare all 12 existing budget decisions. Mean
  outputs and bound hashes must also match the published frozen retiming report.
- Verify source/native/sidecar hashes, unchanged trace hashes and per-query
  numerical errors against the previous public observations. No compression or
  quality thresholds may change in response to the follow-up.
- Time mean-mode **full assessment**, including candidates ultimately refused:
  two warmups, ten paired repetitions, seed-42 randomized Python-first/native-first
  order, one CPU thread. Time same-source FP64 mean aggregation separately on
  all bags. All 18 checkpoints and both known/OOV traces are measured; report
  three-seed medians per convention/order and raw samples. No rejection counts
  as successful coverage. Constructors/hashing/startup are outside these timings.
- Count complete artifact/sidecar bytes as unchanged; C scratch is two record
  arrays proportional to the largest bag, and output plan capacities scale with
  unique query occurrences. No O(N/G) new state, whole-table query scan, hard
  RSS claim, exclusive/cold-cache, GPU, energy or monetary measurement.
- Run exact-fraction and randomized arithmetic differentials, malformed C ABI
  capacity/ID/group/count/norm checks, ASan/UBSan and installed wheel integration.
  Preserve failures and regressions. Publish source/compiler/library/environment
  hashes and actual commands, only metrics/hashes rather than weights/text.

A gain is a full-assessment implementation result at the fixed approximate
contract. It does not imply improved cosine, downstream quality, coverage,
compression or a new n-gram theorem. Direct source or existing raw native queries
may remain faster; their scope and complete memory costs must stay explicit.
