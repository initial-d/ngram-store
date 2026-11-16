# Vectorized native embedding bags, with optional zero-row-staging fusion

The engine now exposes `NativeRows.embedding_bag_array`: a NumPy-only ragged
sum/mean API. FP32 has a dedicated dimension loop that GCC vectorizes without
changing original-row accumulation order, FP64 products/sums or final FP32
conversion. FP16/BF16 retain the existing decoder. Weighted native bags share
the optimization; their existing sum-only semantics remain unchanged.

The default retains bounded row staging because it is faster. All-raw tables
can opt into direct mmap accumulation with `raw_bag_fusion=True`, eliminating
the row matrix while retaining original-ID forwarding and chunk/block counters.
Compressed tables keep chunk staging. No table formats or stored vectors change.

## Default performance on frozen trained tables

Paired **public NumPy/C mean-pooling calls**, including input validation and C
reduction. Excludes tokenization, classifier head, GPU transfers and full models.

| Table, seed 3 | Original rows | Documents/call | Baseline p50 (µs) | Default vectorized p50 (µs) | Paired geometric speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| AG News, uniform | 6,989,612 | 1 | 116.16 | 67.68 | 1.71× |
| AG News, uniform | 6,989,612 | 32 | 2428.91 | 826.23 | 2.98× |
| AG News, zero | 6,989,612 | 1 | 114.27 | 65.87 | 1.73× |
| AG News, zero | 6,989,612 | 32 | 2370.31 | 769.23 | 3.12× |
| DBpedia, uniform | 34,050,161 | 1 | 134.11 | 73.48 | 1.80× |
| DBpedia, uniform | 34,050,161 | 32 | 2956.92 | 972.76 | 3.05× |
| DBpedia, zero | 34,050,161 | 1 | 136.10 | 75.36 | 1.77× |
| DBpedia, zero | 34,050,161 | 32 | 2922.80 | 954.64 | 3.07× |

Both arms read the same existing uncompressed/deduplicated table and each output
is compared byte-for-byte with independently computed source FP64 mean features,
rounded to FP32. All checked features match. Uniform-init byte dedup expands the
table; its inclusion tests the kernel and does not recommend dedup over raw.

- Models/data: the registered AG News/DBpedia classifiers, zero and uniform
  initialization, seed 3; no retraining. 256 official test documents per dataset,
  seed 20261112, original IDs and full lengths. OOV zero removed as in the
  registered `StagedClassifier` protocol. Batches preserve document boundaries.
- Measurement: 2 warmup passes, 10 timed passes, 3 alternating paired rounds;
  independent handles, preallocated outputs, zero native cache, chunk limit
  4096. p50 columns are medians of round p50s. Five CPU jobs ran on distinct
  pinned cores. Ratios are descriptive, not confidence intervals or application
  accuracy/latency claims.
- AMD EPYC 7713, Linux 5.4, Python 3.11.15, NumPy 2.2.6, PyTorch 2.11.0, GCC
  11.4.0, `-O3` without unsafe floating-point or architecture-specific flags.
  Wheel and Makefile native builds use the same optimization level.
  [Compiler diagnostics](tests/vectorization.log) confirm 16-byte vectorization.
- Source commit `4f6fa5a7a7c9e2eb42d5043784d39738150de23b`.
  Existing CPU resources; no GPU, training or paid API. Energy/monetary costs
  unmeasured. [Build and library hashes](build.json), [commands](run-status.json).

## Memory preference and ablations

Fusion alone (`f51f486`) saved row staging but measured **0.95–1.00×** speed.
Dedicated FP32 vectorization plus fusion (`854fd04`) improved the raw-table
calls, but an additional [same-vector-kernel comparison](fusion-ablation.json)
found staged reduction faster on both AG News batch-32 controls. Therefore the
final default is staged/vectorized; fusion is an explicit memory preference.
The comparison reused parent input hashes, rechecked original trace hashes and
compared every output against freshly read source features.

On the checked batch-32 raw calls, fusion reduces the allocated row matrix from
**2,097,152 bytes to zero**. This counter excludes query indexes, FP64 accumulator,
output, caches and total RSS. It is not a total-process memory guarantee.
`last_bag_staging_bytes` reports the actual row-matrix allocation capacity. A
failed optional raw counter-index allocation can fall back to staged reading.

The AG News zero-init Zstd batch-32 control stays about **1.01×** with the final
kernel; decompression still dominates. Both arms still stage 2 MiB. That control
uses 1 warmup, 3 repeats and 2 rounds. All earlier slow/no-gain results remain in
[raw/](raw/); [summary.json](summary.json) indexes all 27 timing cells.

## Use without PyTorch

```python
import numpy as np
from ngram_store import NativeRows

with NativeRows("memory.native/table.men") as memory:
    pooled = memory.embedding_bag_array(
        np.array([1, 42, 1], dtype=np.uint64),
        np.array([0, 2, 3], dtype=np.uint64),
        dtype="float32", mode="mean", chunk_rows=4096,
    )  # two original-ID bags, FP64 accumulation -> FP32 output

# Prefer zero row-matrix allocation on all-raw tables:
with NativeRows("memory.native/table.men", raw_bag_fusion=True) as memory:
    pooled = memory.embedding_bag_array([1, 42, 1], [0, 2, 3])
    assert memory.last_bag_staging_bytes == 0
```

Offsets include the terminal ID count. Empty bags return zero. `padding_id`
excludes that original row from sums and mean denominators. Optional `out`
requires a writable contiguous FP32 array of `(bags, dim)`. Dtype strings are
`float32`, `float16`, `bfloat16`; the caller owns the stored dtype contract.
Existing row/gather APIs and compiled-library default loading remain compatible.
Opt-in fusion requires rebuilding the C library. The C setter is
`meft_rows_set_raw_bag_fusion(store, 1)`.

## Verification and reproduction

[Tests](tests/): 381 local tests passed (149 optional skips), 45 local native
passed (5 optional skips), and 56 remote native/PyTorch bag tests passed without
skips. Local/remote ASan+UBSan passed, including raw fusion on mutated artifacts.
An additional 45-test local native run with `-O3` passed (5 optional skips),
and a fresh Python 3.12 package loaded outside the checkout passed NumPy bag
and mode-switch checks without importing Torch.
Coverage includes all three dtypes, weights, padding, empty bags, forwarding,
odd dimensions, chunk boundaries, compressed caches, output reuse and errors.

The general entry point is
[`tools/benchmark_native_bags.py`](../../tools/benchmark_native_bags.py);
[scripts/](scripts/) retain the exact existing-workspace launches and ablation.
The first two phases computed full table/code hashes. The final default run
explicitly inherited those immutable-input hashes from the parent vector run,
rechecked trace/metadata hashes and every source feature, and records parent
report hashes plus current file stats. It does not describe inherited hashes as
new full-file verification. Omitting `--hashes-from` recomputes full input hashes.

Timing records are gzip-compressed to keep the checkout small. Tables, weights,
query arrays and corpora remain in the experiment workspace.
