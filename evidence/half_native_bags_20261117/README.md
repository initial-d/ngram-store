# Faster native FP16/BF16 embedding bags

Dedicated dimension loops extend the previous FP32 optimization to FP16 and
BF16. GCC 11 vectorizes both at portable `-O3`, without fast-math or CPU-specific
instructions. FP16 subnormals decode as exact mantissa multiples of 2^-24;
integer masks select normal, subnormal and special-value bits without branching.
BF16 expands its original bits directly. Original-ID order, FP64 products/sums,
FP32 output, forwarding, chunking and file formats keep their existing contracts.

## Paired calls on full trained tables

Public NumPy/C mean-pooling, including validation and reduction, on 256 frozen
official-test document traces. The two tables have **6,989,612** and
**34,050,161** original rows, 128 dimensions. Both libraries read identical
uncompressed identity-layout half-format tables. Output matches the stored
half-format source's FP64 mean rounded to FP32 byte-for-byte on every timed call.

| Table, seed 3 | Stored dtype | Documents/call | Baseline p50 (µs) | Final p50 (µs) | Paired speed ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| AG News, zero init | FP16 | 1 | 131.59 | 79.21 | 1.65× |
| AG News, zero init | FP16 | 32 | 2907.33 | 1140.99 | 2.55× |
| DBpedia, uniform init | FP16 | 1 | 142.56 | 87.34 | 1.60× |
| DBpedia, uniform init | FP16 | 32 | 3132.10 | 1384.38 | 2.37× |
| AG News, zero init | BF16 | 1 | 111.48 | 62.44 | 1.78× |
| AG News, zero init | BF16 | 32 | 2210.04 | 578.27 | 3.90× |
| DBpedia, uniform init | BF16 | 1 | 131.73 | 69.64 | 1.83× |
| DBpedia, uniform init | BF16 | 32 | 2766.66 | 791.79 | 3.63× |

These are CPU pooling-call speedups. Tokenization, classifier head, GPU transfer
and complete model inference are outside the timing scope. Low-precision table
creation happens before timing: FP16 reuses the existing frozen FAISS FP16 codes;
BF16 exports the existing FP32 trained rows once with round-to-nearest,
ties-to-even. These conversions have their own approximation error. The kernel
adds no error relative to the stored low-precision table; this experiment makes
no new accuracy claim or claim of training in BF16.

Chunk size 4096, default staged reduction, zero native cache, preallocated output,
2 warmups, 10 repeats, 3 alternating paired rounds, one pinned CPU core per job.
p50 columns are medians of round p50s; speed ratios are geometric means of
paired call ratios, descriptive measurements. Both arms perform identical logical
block/decode work. Hardware: AMD EPYC 7713, Linux 5.4, Python 3.11.15, NumPy 2.2.6,
PyTorch 2.11.0 installed but unused by the timing runner, GCC 11.4.0. Existing CPU
resources; GPU, training and paid APIs unused, energy/monetary costs unmeasured.

The existing compressed AG News FP16/Zstd batch-32 control improves only **1.05×**
(34049.33 → 32409.63 µs p50); decompression remains dominant. It uses 1 warmup,
3 repeats and 2 rounds. Tables, codes and corpora stay outside Git; this evidence
folder contains about 500 KB of reports, logs and scripts.

## Ablation and provenance

The first FP16 specialization still used conditional conversion. On these same
full tables it reached 1.65–1.66× at batch 32; the final branchless conversion
reaches 2.37–2.55×. GCC diagnostics confirm vectorization of the final FP16 loop.
Both initial and final raw timing records are retained in [raw/](raw/).

- Baseline: `633cde3f386663928e2b1ed29be95d2f0b77b41d`, already containing the
  earlier FP32 vectorization; same `-O3` and optional-codec flags in both arms.
- Initial kernel: `5f22b912cbab97081099ac08a2de15b1456122db`.
- Final kernel: `d80f44c6b597a78d29215f12bb284175c1c94c45`.
- [Initial build](initial-build.json) records full code/table hashes, source
  hashes, export rules, table sizes and exact compiler commands.
  [Final build](final-build.json) records the new source/library hashes.
- [Commands](final-run-status.json), [summary](summary.json),
  [initial protocol](scripts/run.py), [final replay](scripts/run_final.py).
  Original trace provenance links to the previously registered AG News/DBpedia
  traces in the remote workspace and is included in every raw report.
- Initial measurements recomputed full table/code hashes. Final replay inherits
  those hashes from the exact initial reports and checks trace/metadata hashes
  and every source feature again. Parent report hashes are recorded; inherited
  checksums are explicitly distinguished from fresh full-file verification.

## Weighted pooling without Torch

```python
import numpy as np
from ngram_store import NativeRows

with NativeRows("memory.native/table.men") as memory:
    output = memory.embedding_bag_array(
        [1, 42, 1], [0, 2, 3], dtype="float16", mode="sum",
        per_sample_weights=np.array([0.5, -2.0, 1.0], dtype=np.float16),
    )
```

Each query occurrence keeps its own weight. Sum mode is required; padding
excludes the row and its weight, including nonfinite padding weights. FP32/FP16
NumPy weight arrays must match the stored dtype; Python numeric sequences are
cast to that dtype. BF16 weights use NumPy uint16 bit patterns. Noncontiguous
and opposite-endian arrays are converted to contiguous LE buffers. Empty bags,
chunking and preallocated writable FP32 `out` remain supported; output cannot
overlap the buffers passed to C. Torch's existing weighted bag adapter benefits
from the same kernel automatically.

## Validation

[Logs](tests/): 383 local tests (149 optional skips); 47 local native tests
(5 optional skips); 58 remote native/PyTorch bag tests with no skips; local and
remote ASan/UBSan; a fresh installed platform wheel tested outside the checkout
for all three weighted dtypes without importing Torch.

All **65,536 bit patterns of each half format** match the scalar control at O3,
including nonfinite patterns, with independent NumPy checks of finite values.
Coverage includes reversed physical rows/forwarding, compressed/staged/fused
paths, chunk boundaries, padding, repeated IDs, empty bags and weighted-array
validation. The initial remote test transfer omitted `examples/native_gather.c`;
its failed log is preserved. The complete transfer and final rerun passed all
58 tests. This was a transfer error, resolved before the final verification.

```bash
python -S evidence/half_native_bags_20261117/verify.py
MEFT_NATIVE_TEST_FLAGS=-O3 python -m unittest tests.test_native_rows tests.test_mapped_embedding_bag
python tools/check_native_sanitizers.py
```

The verifier checks retained timing/provenance consistency. Re-running actual
queries requires the frozen external tables and traces described by the scripts.
