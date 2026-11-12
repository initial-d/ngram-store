# Identity format on real trained Qwen memory

MEFTNAT1 v3 removes a redundant 8-byte forwarding entry when every physical row
already has its original ID. The fixed map was 6.25% of these 32-dimensional
FP32 payloads. All original trained rows, zero sentinels and budget padding are
retained; codec payloads and block sizes are unchanged.

This experiment reuses the six 20,877-edit exact-key checkpoints from the
[independent-row study](../ngram_identity_evidence_20261107/README.md) and the
[original native offload audit](../qwen_native_identity_evidence_20261109/README.md).
It performs no new training or GPU work.

## Storage and query result

The default 32-row v3 table plus metadata now uses about **5.4–5.5% fewer
bytes than the unchanged FP32 row payload**, where v2 expanded slightly.
The format-only table reduction is exactly `8 * nrows` in every matched layout.
These are modest lossless file savings; there is no demonstrated speed advantage.

Storage columns below use seed 3; warm p50s are medians of the three per-seed
p50s. v2 counts the table; v3 includes the table and metadata, so the exact
format difference is not the difference of those two columns.

| Model | Original FP32 rows | v2 table | v3 table + metadata | Exact table saving | v3 saving vs raw payload | Old v2 / new v3 warm p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 74.356 MB | 74.917 MB | 70.271 MB | 4,647,280 B | 5.49% | 0.0630 / 0.0619 ms |
| Qwen3.5-4B | 73.514 MB | 74.123 MB | 69.529 MB | 4,594,640 B | 5.42% | 0.0629 / 0.0643 ms |

Larger blocks improve file capacity but amplify misses. These are seed-3 v3
row-query p50s, with no application cache versus the 1 MiB row cache after
warmup. Both use a warm OS page cache. All three library/format controls and
unfavorable measurements remain in the archive.

| Model | Rows per block | v3 table + metadata | Zero-cache p50 | Warm row-cache p50 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 32 | 70.271 MB | 1.15 ms | 0.0634 ms |
| Qwen3-0.6B | 128 | 69.201 MB | 3.62 ms | 0.0615 ms |
| Qwen3-0.6B | 512 | 69.079 MB | 14.16 ms | 0.0624 ms |
| Qwen3-0.6B | 2048 | 69.050 MB | 50.48 ms | 0.0615 ms |
| Qwen3.5-4B | 32 | 69.529 MB | 1.14 ms | 0.0643 ms |
| Qwen3.5-4B | 128 | 68.460 MB | 3.61 ms | 0.0643 ms |
| Qwen3.5-4B | 512 | 68.339 MB | 14.11 ms | 0.0610 ms |
| Qwen3.5-4B | 2048 | 68.309 MB | 49.57 ms | 0.0639 ms |

The largest blocks reach roughly 7% saving against raw rows, but zero-cache
queries take around 50 ms. They are not a default recommendation. Seed-3
dense/raw-mmap row-query p50s are disclosed in `summary.json`; native has no
speed advantage over those controls. Open time is noisy, and no faster startup
claim is made. Anonymous RSS remains dominated by runtime/scratch/cache;
removing the map reduces file bytes and the queried file-backed working set,
not necessarily whole-process or whole-model RSS by `8 * nrows`.


## Matched controls and compatibility

Each model has seeds 3/7/11. All six tables use the same 64 chronological
MultiCounterFact test-prefix row requests as the earlier audit. At seed 3,
32/128/512/2048-row blocks are all disclosed; the other seeds check 32-row
blocks. Every layout is packed both ways and reconstructed in full. Normalized
block descriptors and codec payload hashes match exactly. Legacy v2 exports
at 32 rows match the previous artifact hashes, and the actual old library
reconstructs every row. That library rejects v3 explicitly.

For each block/cache setting, fresh workers compare old-library/v2,
new-library/v2 and new-library/v3. Both zero cache and a 1 MiB row cache are
tested. Three warmup traces and 20 timed traces use identical original queries;
every first-trace output hash is checked. Seed-3 dense and raw-mmap controls
complete the 76 CPU workers. Order alternates across configurations, but there
is only one process per reader/configuration: this is not a replicated process
latency study. Raw samples, wall/CPU time, open time and Linux anonymous/file
RSS are preserved in [summary.json](summary.json) and the reports under `results/`.

All runs use one thread and one CPU affinity on a shared Linux host. Full
reconstruction warms the OS page cache before workers. Open time includes
metadata, hashing and adapter work; it is not bare C-open latency. No disk-cold,
cgroup-pressure, whole-model startup or production serving result is claimed.
Hardware, compiler, source SHA and cost assumptions are in [build.json](build.json)
and each report. No rented compute, data downloads or paid API was used;
energy and monetary costs are unmeasured.

## Interpretation

This is a fixed format-overhead reduction, with no numeric approximation or
new compression theory. It resolves the redundant identity-map cost identified
by the previous audit. It introduces no DATrie/mark-compact storage claim.
Reordered, deduplicated and representative exports still need explicit maps;
`implicit_identity=False` remains available for old compiled readers.

Byte preservation is established for all rows and measured row requests.
The previous 512 complete CUDA-prefix logit pairs used v2, and are not counted
as new v3 model measurements. CPU fixtures exercise complete Qwen reopen on v3,
but this archive does not rescore native v3 on full MultiCounterFact or measure
new GPU memory/latency. Original trained vector bytes, query arrays and model
weights remain in their source runs; they are not shipped in this archive.

## Reproduce and audit

Check the published provenance, assertions and arithmetic without dependencies:

```bash
python3 -S docs/identity_format_evidence_20261109/verify.py
```

With an original verified fixture and both compiled libraries, rerun from code
commit `3498f99b3aea998df67c7ac9cc446bc0cf7eba51`:

```bash
python -m tools.benchmark_identity_format \
  --fixture /path/to/verified-trained-fixture \
  --output /path/to/fresh-results \
  --native-library /path/to/v3/libmeft_rows.so \
  --old-library /path/to/v2/libmeft_rows.so \
  --block-sizes 32,128,512,2048 --cache-sizes 0,1048576 \
  --warmup 3 --repeat 20
```

The exact per-worker commands and execution order are in each `schedule.json`;
[run.sh](run.sh) records the full six-table run. Runtime snapshots are archived
under `runtime/`. The stdlib verifier does not reconstruct withheld vectors or
rerun timings; it checks the archive against prior source manifests and verifies
all reported layout, output-hash, storage and timing arithmetic.
