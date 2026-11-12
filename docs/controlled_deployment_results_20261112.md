# Completed controlled deployment experiments

All registered Qwen, CPU and public-classifier deployment grids finished on
2026-11-11 at 15:56 UTC. The final application audit exited successfully.
The experiment preserves unsuccessful capacity gates, slower queries and actual
CUDA allocator OOMs alongside successful controls.

| Grid | Audited completion |
| --- | --- |
| Qwen3-0.6B / Qwen3.5-4B | 108 paired blocks, 864 passing reports, 648 source/native contrasts |
| CPU fixed row layouts | 1,044 reports and 288 frozen capacity-gated choices |
| CPU canonical cache/concurrency | 5,040 reports |
| AG News | 6 source models, 6,989,612 rows, 7,600 official test documents; 1,980 GPU cells: 1,620 passed, 360 allocator OOMs |
| DBpedia | 6 source models, 34,050,161 rows, 70,000 official test documents; 1,980 GPU cells: 1,476 passed, 504 allocator OOMs |

Each public application has 88 candidate jobs per source model and 180 paired
GPU blocks. CPU quality checks use the full official test sets; 120 successful
full-test GPU checks compare predictions and logits with CPU results. Allocator
caps apply to PyTorch CUDA allocations and exclude host RAM and staging.

## Capacity and actual query costs

The table compares calibration-selected exact-byte storage with direct FP32
host access, including complete model metadata. Savings span three trained
seeds. Query ratios are observed ranges across the same-device/core pairs,
three cohorts, batches 1/32 and five allocator budgets, with a balanced geometric
mean in parentheses. They are descriptive measurements, not confidence bounds.

| Dataset / initialization | Exact-byte selected method | Complete model GiB, direct → selected | Saving | Host query p50 slowdown |
| --- | --- | --- | --- | --- |
| AG News / zero | byte dedup + Zstd | 3.438 → 0.552 | 83.93–83.95% | 33.5–66.5× (42.2×) |
| AG News / uniform | generic Zstd | 3.438 → 3.153 | 8.28–8.29% | 29.3–78.8× (48.1×) |
| DBpedia / zero | byte dedup + Zstd | 16.754 → 2.841 | 83.04% | 31.4–52.1× (41.6×) |
| DBpedia / uniform | generic Zstd | 16.754 → 15.313 | 8.60% | 37.3–57.7× (46.4×) |

These methods preserve original vector bytes and held-out classifier outputs,
but their measured complete host queries are substantially slower. This is
capacity evidence, not deployment dominance or a DATrie-specific improvement.
Initialization materially changes exact deduplication opportunities.

The uniform branches also select native Zstd over existing FP16 codes at the
registered 0.5 pp budget. Those choices use about 1.626 GiB on AG News and
7.900 GiB on DBpedia. Their host p50 is about 4.9× slower than direct FP16 on
balanced geometric means; FP16 retains its own approximation error. Full/hash
classification differences are small. The original DBpedia uniform LR3 result
and the separately registered LR10 training-participation sensitivity remain
separate; the secondary result does not replace the primary selection.

Qwen source/native pairs preserve decoded codec bytes, prompt hashes, cached
logit hashes and generated-token hashes. Libdeflate complete-inference ratios
range from 0.979 to 1.021 across model/codec/batch summaries, supporting comparable
complete latency. Independent row packets and LZ4 fail both capacity gates;
Zlib and Zstd qualify in 35/72 choices each but remain slower on the held-out
complete-token CPU query. The experiments establish explicit tradeoffs rather
than universal superiority over FAISS.

## Inspect and reproduce the numeric evidence

- [Complete public-application summary, including query costs](controlled_application_results_20261111.json)
- [Qwen byte/output identity and paired latency](controlled_qwen_gpu_results_20261111.json)
- [CPU capacity gates and cache/concurrency summary](controlled_cpu_results_20261111.json)
- [Training-participation sensitivity](training_participation_sensitivity_20261111.json)
- [Evidence release](https://github.com/initial-d/ngram-store/releases/tag/controlled-deployment-evidence-20261112)

Download `manifest.json` and its capsules into one directory, then verify:

```bash
python -S -m tools.package_controlled_evidence --verify artifacts/controlled-evidence
```

The verifier checks every capsule, member and original audit hash link without
extracting archives. Prediction NPZ files contain only logits, labels and NLL;
the NumPy payload audit independently checks their quality arithmetic. Runtime
snapshots, frozen jobs, calibration predictions, raw timing samples, commands
and environment versions accompany the reports. Trained vectors, lexical keys,
corpora and checkpoints are excluded. Hash verification does not rerun training
or independently reconstruct source rows without those excluded fixtures.
Energy and monetary costs were not measured.

To regenerate the checked-in application summary from the downloaded audit:

```bash
python -m tools.summarize_controlled_application \
  --audit artifacts/controlled-evidence/cpu-audit/application-results.json \
  --output docs/controlled_application_results_20261111.json
```
