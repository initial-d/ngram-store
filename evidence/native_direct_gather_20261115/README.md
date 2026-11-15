# Direct native gathers on existing trained tables

All-raw native tables now bypass block sorting and copy directly from their
contiguous physical payload after validating every requested original ID and
forwarding entry. Distinct-block counters use a reusable query-sized hash;
ordered queries avoid that hash. Allocation failure falls back to the grouped
reader. Original IDs, aliases, row bytes, file formats and existing C ABI remain
compatible. Cache budgets do not cause raw payload copies into a decoded cache.

## Measured result

Paired CPU **row gathers**, not complete classification or LLM inference:

| Trained table | Original rows | Baseline p50 (µs) | Direct p50 (µs) | Paired geometric speed ratio |
| --- | ---: | ---: | ---: | ---: |
| AG News, uniform, seed 3 | 6,989,612 | 35.34 | 22.28 | 1.59× |
| AG News, zero, seed 3 | 6,989,612 | 32.90 | 21.02 | 1.57× |
| DBpedia, uniform, seed 3 | 34,050,161 | 45.68 | 27.17 | 1.65× |
| DBpedia, zero, seed 3 | 34,050,161 | 43.30 | 26.05 | 1.63× |

Each arm reads the **same existing deduplicated, uncompressed artifact**; every
warmup and timed output is compared byte-for-byte with the original FP32 codes.
This reduces query overhead without changing stored capacity or model weights.
The previously measured [table savings and GPU host path](../host_query_profile_20261112/README.md)
remain separate measurements. Uniform-init byte dedup does not save capacity;
including that layout here tests the reader and does not recommend it over raw.

- Source implementation: `4a89d3c55b77ce36830872271cdf726324ab2e49`.
- Data/models: existing registered AG News and DBpedia classifiers, zero and
  uniform initialization, seed 3. 256 official test documents per dataset,
  sampled without replacement with seed 20261112; OOV ID zero removed as in
  `StagedClassifier`. The request IDs are never reordered or replicated.
- Measurement: 2 warmup passes, 10 timed passes, 3 paired rounds; baseline and
  candidate alternate per request. Independent handles, preallocated inputs and
  outputs, zero native cache, one pinned CPU core per model. Both use the same
  optional-codec build. p50 columns are medians of the three round p50s.
- Environment: AMD EPYC 7713, Linux 5.4, Python 3.11.15, NumPy 2.2.6,
  PyTorch 2.11.0, GCC 11.4.0. No GPU, retraining or paid API was used.
  Energy and monetary costs are unmeasured.
- Scope: warm file-cache row lookup after addressing. Four jobs ran in parallel
  on distinct cores; ratios are descriptive, not confidence intervals. This
  does not measure quantizer decoding, reductions, transfer or full requests.

## Conditional XOR optimization and negative control

Previous-row XOR stream queries now reconstruct through the highest requested
physical row when a full decoded block will not enter the cache. Complete stream
decompression and validation still run; full-block cache fills reconstruct the
entire block. `NativeRows.last_xor_bytes` / `meft_rows_last_xor_bytes()` distinguish
this work from full decoded bytes. Chunked gathers and embedding bags aggregate
both counters.

The first controlled run (`1e82e5e`) reused the saved 32 official MultiCounterFact
query traces of two trained Qwen memory tables (seed 3), at block sizes
16/64/256/1024. Across all eight cases, the speed ratio stayed **about 1.00×**.
Only two cases reconstructed any previous-row XOR at all; their XOR work fell
by 13% and 33%, but stream decompression dominated. This is a conditional work
reduction, not a demonstrated Qwen speedup. Plain/raw and independent row-packet
codecs are unaffected by this predictor change.

## Checks and reproduction

[Tests](tests/): local suite 379 tests passed (149 optional skips); local native
suite 43 passed (5 optional skips); remote native suite 43 passed without skips,
including PyTorch and optional LZ4/Zstd/libdeflate. ASan/UBSan passed locally and
remotely, including raw direct reads, malformed streams and cache growth.

[summary.json](summary.json) indexes all twelve original timing records in
[raw/](raw/), compressed with gzip to keep the checkout small. Records include
commands, file and library SHA256s, source commit, environment, raw per-request
timings and logical-work counts. [build.json](build.json) records compiler,
headers, library hashes and baseline flags. Weights, tables, query arrays and
corpora are retained in the user's experiment workspace and are not added here.

On the existing registered experiment workspace, compile both libraries using
`build.json`. The later baseline disables both optimizations with
`-DMEFT_DISABLE_RAW_GATHER -DMEFT_DISABLE_XOR_PREFIX`; the first Qwen control
disables only XOR prefixes. Run:

```sh
PYTHONPATH=/llm-align/duyimin/ngram-sparse-xor-20261115/runtime \
  /llm-align/duyimin/ngram-counterfact-capacity-20261106/env/bin/python \
  evidence/native_direct_gather_20261115/scripts/run_raw.py \
  --work /llm-align/duyimin/ngram-sparse-xor-20261115 \
  --commit 4a89d3c55b77ce36830872271cdf726324ab2e49
```

The general measurement entry point is
[`tools/benchmark_native_gather.py`](../../tools/benchmark_native_gather.py).
It rejects changed trace hashes, invalid IDs, table/code dimensions, byte
mismatches or changed decode work between the two arms. Rebuilding the trained
artifacts requires the original training protocol and data; these saved results
alone do not reproduce a new model-quality experiment.
