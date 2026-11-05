# Native residual-bound query planning

The C compiler replaces Python group dictionaries, centered residual-bound loops,
strict-domain checks and reduction-plan construction in the opt-in
`BudgetedBagRows` path. ID counting, row reading/reduction and the remaining
norm/rounding work retain their existing implementations. The default strict
certificate, old ABI, native table and residual-sidecar formats are unchanged.

On 2026-11-05, the author reran the **same 18 finite enwik8-prefix SGNS
checkpoints**, orders 4/8/12 and seeds 3/7/11 under legacy/average conventions.
No new training, downloads or GPU access. These are stored n-gram component
sum/mean queries, with missing n-gram types omitted equally in each reference;
they are not complete model-output or LLM memory/gate benchmarks.

## Full assessment latency

One CPU thread, two warmups and ten paired repetitions with seed-42 randomized
Python/native order. Each known/OOV trace contains 512 queries. Values below are
three-seed medians of each trace's timing p50; speedup is the median of paired
per-seed ratios. Timing includes candidates rejected by budget gates and all
assessment work, while constructors, hashing and startup are outside timing.

| Convention / order | Known Python → C (ms) | Known speedup | OOV Python → C (ms) | OOV speedup | Source FP64 OOV (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy / 4 | 97.40 → 52.33 | 1.86× | 110.05 → 62.09 | 1.77× | 10.81 |
| Legacy / 8 | 128.00 → 54.78 | 2.33× | 120.60 → 63.71 | 1.89× | 11.20 |
| Legacy / 12 | 132.44 → 55.03 | 2.40× | 120.95 → 63.87 | 1.89× | 11.29 |
| Average / 4 | 97.85 → 52.56 | 1.87× | 106.96 → 58.79 | 1.82× | 10.84 |
| Average / 8 | 128.21 → 54.79 | 2.34× | 120.54 → 63.51 | 1.90× | 11.27 |
| Average / 12 | 132.55 → 55.04 | 2.42× | 121.33 → 63.81 | 1.90× | 11.26 |

The complete bounded path still takes roughly **4.4–5.7×** the source FP64
control at these group medians. Source controls retain all FP32 source rows and
compute FP64 means; the bounded backend retains the compressed table and
source-residual scalars. This implementation gain does not establish superiority
over dense/raw serving or FAISS. The shared CPU was not exclusive: average/order
4/seed 7 has much slower samples (OOV Python 346.87 ms, native 145.57 ms). All
samples are retained, and that checkpoint participates in the predeclared
three-seed medians; no best-run substitution or retuning was performed.

## Frozen correctness and capacity

For all 18 checkpoints and both traces, sum and mean output, projection/rounding/
absolute/relative-bound and strict-domain arrays match Python **byte-for-byte**.
All 12 existing relative-L2 budget decisions match. Mean output and absolute-bound
hashes also match the [previous public retiming](../budgeted_bag_evidence_20261104/retiming.json).
Trained source, native artifact, sidecar and trace hashes are verified before use.
Thus the original per-query error observations remain applicable through identical
outputs/bounds; this is not new downstream-quality evidence.

Complete **model and sidecar** bytes are unchanged; the C library is newly built
and its hash is reported separately. C scratch is two record arrays sized to the
largest observed bag. Plan outputs scale with unique query occurrences. This
introduces no extra full-table/group state or whole-table query scan; there is
no new RSS, hard-memory-cap, cold-cache, energy or monetary measurement.
The earlier fixed-budget coverage and complete-capacity results are inherited,
not additional gains. Approximation remains explicit and source-specific.

## Inspect and reproduce

- [Frozen protocol](../native_bounded_query_protocol_20261105.md), measured source
  commit `56df5865dd58ae5a396ffa2910b8c953e9511782`.
- [Raw samples, commands, CPU/compiler/library/source hashes](report.json),
  [checkpoint log](benchmark.log), [derived summary](summary.json), and
  [environment and local commands](environment.json).
- [Remote reproduction](reproduction/run.sh) uses cached user-owned inputs and a
  `git archive` of the measured commit. Original text, vectors and sidecars are
  not redistributed. No GPU inspection or process changes are required.
- [Tests](tests/): local suite 197 tests, 100 optional-dependency skips; remote
  ten bounded-query tests pass; local/remote ASan+UBSan pass; locally built
  macOS wheel passes outside-checkout NumPy-only native-compiler integration.
  Exact-fraction oracles and randomized arithmetic tests are correctness tests,
  not synthetic-vector compression evidence. Released v0.4.0 wheels do not
  contain this addition; no new release or PyPI upload is claimed.

The frozen report's `command` field reconstructed `sys.argv` and therefore loses
the original `-m` invocation. The exact executable module command is preserved
in `environment.json` and `reproduction/run.sh`; the reporting code now captures
`sys.orig_argv` for subsequent runs. The measured report remains unedited.

Offline public verification requires only Python's standard library:

```bash
python docs/native_bounded_query_evidence_20261105/verify.py
```

It validates every published timing median/ratio, all 18 source/trace/output/
bound/budget/capacity contracts against the previous public report, and summary
and evidence hashes. Rerunning queries still requires the source-linked cached
trained inputs. This is author-run evidence, not independent adoption.
