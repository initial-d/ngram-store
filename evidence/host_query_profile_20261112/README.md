# Where host query time goes on the AG News classifier tables

The [controlled deployment grids](../controlled_deployment_results_20261112.md)
found native host lookups 29–79× slower than direct FP32 on the AG News and
DBpedia classifier tables. This profile asks why, on a local reproduction of the
AG News source model.

## Answer

Almost all of the time is block decompression. The bundles in that grid used
256-row blocks (128 KiB per block at 128 × FP32), and a document's n-gram IDs are
spread across the table because the registry is sorted by key bytes. Each lookup
of about 130 IDs decodes up to 130 blocks, roughly 16 MiB, to return 65 KiB.

On the zero-initialized table, nearly all of the saving comes from duplicate rows,
which need no compression. Deduplication alone saves 85.8% (86.6% with Zstd)
and brings lookups to 1.2× raw mmap.

| Layout (zero init) | Saving vs FP32 | p50 per document | vs raw mmap |
| --- | ---: | ---: | ---: |
| raw mmap (`np.load(..., mmap_mode='r')`) | 0% | 14.3 µs | 1.0× |
| dedup + Zstd, 256-row blocks (grid selection) | 86.6% | 5,095 µs | 356× |
| dedup + Zstd, 16-row blocks | 86.5% | 514 µs | 36× |
| dedup + Zstd, 4-row blocks | 86.1% | 234 µs | 16× |
| **dedup, uncompressed** | **85.8%** | **17.3 µs** | **1.2×** |
| generic Zstd, 256-row blocks | 46.7% | 3,685 µs | 258× |

On the uniform-initialized table there are almost no duplicate rows. Zstd saves
8.5% (the grid reported 8.3%) at 427× the lookup time; deduplication adds a row
map and grows the file 1.6%. Raw storage is the right choice there.

| Layout (uniform init) | Saving vs FP32 | p50 per document | vs raw mmap |
| --- | ---: | ---: | ---: |
| raw mmap | 0% | 14.1 µs | 1.0× |
| generic Zstd, 256-row blocks (grid selection) | 8.5% | 6,014 µs | 427× |
| dedup, uncompressed | −1.6% | 18.2 µs | 1.3× |
| native, uncompressed, no dedup | −0.02% | 17.9 µs | 1.3× |

Timings above call the C gather with NumPy buffers. Two other findings:

- **Python binding overhead.** The grid used `NativeRows.rows(ids.tolist())`,
  which checks each ID in Python and builds a ctypes array. With compressed
  blocks this is invisible (<1%); on uncompressed tables it doubles lookup time
  (37.5 µs vs 17.3 µs). The new `NativeRows.gather_array` takes a NumPy array and
  runs at 19.0 µs.
- **Batching.** Fetching 32 documents in one call instead of 32 calls cuts total
  time 1.25–1.6× on compressed layouts (shared blocks are decoded once) and does
  nothing for uncompressed ones.

The grid's ratios (29–79×) are smaller than the 356–427× here because each grid
sample also includes PyTorch staging and GPU transfer, which both arms pay.

## Setup

- Data: official AG News, prepared with `tools.large_ngram_application prepare`
  (word 1–4-grams, seed-42 90/10 split): 6,989,612 rows, 108,000 training and
  7,600 test documents. Same row count as the grid.
- Model: [`scripts/train_cpu.py`](scripts/train_cpu.py), a CPU copy of
  `tools.large_ngram_application.train` with exact addressing, seed 3, 10 epochs,
  batch 512, lr 10, zero or uniform(−1/128, 1/128) initialization. Test accuracy
  89.7% (zero init). 12.6% of rows are unique on the zero-init table.
- Layouts: [`scripts/build.py`](scripts/build.py) with `pack_native`, 512-byte rows,
  `block_rows=256`, `predictor='none'`, optional `max_block_bytes` (8192 → 16 rows,
  2048 → 4 rows). Zstd 1.5.7.
- Measurement: [`scripts/bench.py`](scripts/bench.py). For each of the 7,600 test
  documents, drop OOV ID 0 (mean 130 IDs left), fetch the rows, and sum them in
  FP64, as `StagedClassifier` does before GPU transfer. Two passes over all
  documents; the second (warm page cache) is reported. No application cache.
- Machine: Apple M5, macOS 26.6.2, Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0 (CPU),
  source commit `d919a44` plus the `gather_array` change in this commit.

Raw results with p95 and totals: [`results/zero-init.json`](results/zero-init.json),
[`results/uniform-init.json`](results/uniform-init.json).

## Limits

This is an exploratory profile, not a pre-registered experiment. It uses one seed,
one learning rate and one machine, and runs the CPU part of the host path only.
The original trained tables were not available, so absolute sizes differ slightly
from the grid (86.6% here vs 83.9% complete-model saving there, which also counts
the registry and classifier). The scripts read `/tmp` paths as run.

## Follow-up

- Re-run the controlled CPU and GPU host cells with `dedup + uncompressed` and
  `gather_array` on the original tables.
- Let the layout advisor ([roadmap](../../docs/roadmap.md) item 2) choose between
  raw, dedup-uncompressed and compressed layouts from the duplicate ratio and a
  latency budget.
