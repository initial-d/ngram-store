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

## Confirmation on the original tables

The same check on the twelve trained tables of the controlled grid (AG News and
DBpedia × zero/uniform initialization × seeds 3/7/11), on the grid machine
(AMD EPYC 7713, 8× A800). The grid's own numbers for the compressed layouts are
taken as published and not re-measured, except one model as a cross-check.

CPU lookup, same per-document operation as above, 5,000 test documents per model
(seeded sample), one pinned core per model:

| Tables | Dedup, uncompressed: saving | Selected grid layout: saving | Lookup vs raw mmap |
| --- | ---: | ---: | ---: |
| AG News, zero init (3 seeds) | 85.80% | 86.58–86.59% (dedup + Zstd) | 1.74–1.76× |
| DBpedia, zero init (3 seeds) | 84.95% | 85.68–85.69% (dedup + Zstd) | 1.44–1.77× |
| AG News, uniform init | −1.59% | 8.54–8.55% (Zstd) | 1.81–1.87× |
| DBpedia, uniform init | −1.59% | 8.87–8.88% (Zstd) | 1.85–1.88× |

On AG News zero seed 3, the grid's layout and library measured 361× raw here;
the old binding on the uncompressed layout 3.8×, `gather_array` 1.8×.

Full GPU host path, using the grid's own `StagedClassifier`, request set (first 256
test documents), 2 warmup and 3 timed passes, one A800 per model:

| Model (seed 3) | Batch 1 vs raw | Batch 32 vs raw | Logits identical |
| --- | ---: | ---: | --- |
| AG News, zero | 1.18× | 1.14× | yes |
| AG News, uniform | 1.16× | 1.14× | yes |
| DBpedia, zero | 1.19× | 1.23× | yes |
| DBpedia, uniform | 1.18× | 1.09× | yes |

The grid reported 29–79× for the selected compressed layouts on this path. Every
table's uncompressed dedup layout returned bytes identical to the source on the
checked documents.

A long-running `gpu_use.py` process from another user held 756 MiB on each GPU and
about 2.5 CPU cores throughout; GPU utilization stayed at 0–3%, and our processes
were pinned to other cores. Results: [`original-tables/`](original-tables/).

## Setup (local reproduction)

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

This profile was not pre-registered. The GPU confirmation covers seed 3 of each
dataset × initialization; the CPU confirmation covers all twelve tables. Savings
here are table-file savings; the grid's complete-model figures also count the
registry and classifier. Scripts record the paths they ran with.

## Follow-up

- Let the layout advisor ([roadmap](../../docs/roadmap.md) item 2) choose between
  raw, dedup-uncompressed and compressed layouts from the duplicate ratio and a
  latency budget.
