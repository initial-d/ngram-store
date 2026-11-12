# Evidence archive

Each experiment has a protocol written before the run, the source commit, the
results, and in most recent cases a `verify.py` that recomputes the published
numbers from the archived data. Directory names end with the run date.

Run a verifier with `python -S evidence/<dir>/verify.py`; CI runs the ones listed
in [`ci.yml`](../.github/workflows/ci.yml).

Files listed in a `manifest.json` are hash-locked and never edited. That is why a
few one-line redirect pages (`news_shift_app.md`, `cotraining_groups.md`, …) sit
here: links in locked READMEs still point to them, and they forward to
[`docs/guides/`](../docs/guides/). When this directory moved out of `docs/` on
2026-11-12, the only locked files that changed were one path line in each of
`budgeted_bag_evidence_20261104/verify.py` and
`native_bounded_query_evidence_20261105/verify.py`, with their manifest hashes.

---

## 1. Does exact n-gram identity beat hash buckets?

| Report | Result and boundary |
| --- | --- |
| [Independent n-gram rows vs shared hash buckets](ngram_identity_evidence_20261107/README.md) · [design](ngram_identity_experiment.md) | 24 final models, equal row budgets, three seeds. At 20,877 edits, independent rows improve H by 0.41/1.24 pp vs matched-coverage hash and 2.29/4.05 pp vs native open-vocabulary hash (0.6B/4B), with positive pointwise 95% intervals. Small-grid results are mixed; frozen-backbone editing only |
| [Complete trained-memory application study](memory_application_evidence_20261110/README.md) | 108 SGNS and six Qwen checkpoints, all 261 codec candidates each. Full-key memory beats matched adapters by 7.02/5.39 pp; some PQ tables expand and native CPU queries are slower |
| [Controlled deployment grids](controlled_deployment_results_20261112.md) · [protocol](controlled_memory_deployment_protocol_20261111.json) | 864 Qwen GPU reports, full AG News/DBpedia tests, 6,084 CPU reports. Capacity savings and query slowdowns reported together; no universal full-key advantage over hashing |
| [MultiCounterFact small grid](counterfact_small_evidence_20261107/README.md) · [interfaces](counterfact_capacity_evidence_20261106/README.md) · [design](counterfact_capacity_experiment.md) | Complete 1,024-edit three-seed grid: no stable large-table gain; BM25 is strongest. Native bundles preserve outputs but expand |
| [Held-out Qwen branch and row identity](qwen_quality_evidence_20261106/README.md) | Memory-off raises test PPL 11.19%; head constants only 0.0416% (interval crosses zero). This checkpoint does not show a need for distinct rows |
| [Eight-GPU Qwen training, codecs, serving](qwen_gpu_grid_evidence_20261106/README.md) | 12 checkpoints, 63 codec comparisons, 69 CUDA workers. 1 MiB row cache speeds warm native paths 1.33–1.39× with exact outputs; large branches fail held-out necessity gates |
| [Joint decoder/memory training](qwen_joint_evidence_20261106/README.md) | Full tables lose to trained constants and memory-off in all three seeds; same-model row identity has a smaller positive effect |

## 2. Do prefix/suffix (tree) neighbours learn similar vectors?

| Report | Result and boundary |
| --- | --- |
| [Tree-sharing quality and training mechanism](tree_node_quality_evidence_20261109/README.md) · [protocol](tree_node_quality_protocol_20261109.md) | 18 checkpoints × 204 candidates; 36 trajectories. Order-12 .99/.99 native n-gram-only saving rises from 1.34% at 2k steps to 41.49% at 32k steps. FAISS and precomputed controls keep their limits |
| [Duplicates are exactly identical incidence](incidence_duplicates_20261113/README.md) | On the zero-init AG News table, grouping rows by bag incidence and by bytes gives the same 883,125 groups |
| [Sharing stored without compression](structure_dedup_20261113/README.md) | Joint `.99/.99` rows are byte duplicates, so dedup + uncompressed keeps 42.4% of the 46.0% saving at 32k steps with lookups at ~2× raw instead of ~125× |
| [Conditional prefix/suffix similarity](conditional_similarity_evidence_20261109/README.md) · [protocol](conditional_similarity_protocol_20261109.md) | 27 character models: some parent–child pairs exceed .99 cosine beyond matched unrelated controls; frequency, coverage and initialization matter |
| [Original DATrie / mark-compact ablations](structural_native_evidence_20261030/README.md) | Measures DATrie and approximate joint sharing separately; exact mode finds no identical rows to merge |
| [Large-table structural applicability](large_structural_evidence_20261031/README.md) | 1.85M rows: extra structural savings below 0.63%; broad queries slower |
| [FP16/INT8 + structural controls](quantized_structural_evidence_20261031/README.md) | Approximate savings persist after quantization; 4-order GRU INT8 regresses |
| [Trained n-gram similarity](trained_ngram_similarity_20261018.md) · [distortion curves](approximate_ngram_compression_20261018.md) · [cosine merge data](cosine_merge_evidence_20261018/) · [order sweep](ngram_order_native_20261018.md) | Early reference measurements (orders 4/8/12, three seeds) |
| [Legacy exporter roundtrip audit](audits/legacy-roundtrip-20261024/README.md) · [merge checks](legacy_merge_check_20261023.md) | Corrected structure + cosine + norm-ratio guards on real `.z` export/reopen |

## 3. What does lossless native storage cost and save?

| Report | Result and boundary |
| --- | --- |
| [Identity format v3](identity_format_evidence_20261109/README.md) · [real Qwen native offload](qwen_native_identity_evidence_20261109/README.md) | Removes 8 bytes per row; 5.4–5.5% saving vs raw FP32 at 32-row blocks. No query speed advantage across 76 CPU workers |
| [1.85M-row CPU capacity](cpu_capacity_evidence_20261030/README.md) · [sparse blocks](sparse_blocks_evidence_20261030/README.md) | Smaller blocks cut native miss latency 16–27×; raw mmap remains faster |
| [Large FP16/INT8 capacity](quantized_capacity_evidence_20261031/README.md) | FP16 native saves 52.29% vs raw FP32; INT8 native expands vs raw INT8 |
| [Host query profile on AG News](host_query_profile_20261112/README.md) | Time is block decompression. On the twelve original tables, dedup without compression saves 85.0–85.8% (vs 85.7–86.6% with Zstd) and the GPU host path runs 1.09–1.23× raw with identical logits; on uniform-init tables raw is best. `ngram-store plan` predicts these layout sizes exactly |
| [Native query/cache controls](native_query_evidence_20261030/README.md) · [decoded-block cache](native_cache_20261022.md) ([data](native_cache_evidence_20261022/)) | Cache sizing and hit/miss behaviour |
| [XOR blocks](native_xor_compression_20261020.md) ([data](native_xor_evidence_20261020/)) · [streaming export](streaming_export_20261021.md) ([data](export_evidence_20261021/)) · [first native pilots](native_ngram_evidence_20261018/) | Codec and export-path development |
| [Embedding bags](embedding_bag_evidence_20261025/README.md) · [bag ID compaction](bag_compaction_evidence_20261026/README.md) | Native bag queries on trained memory |
| [Held-out lossless compression](heldout_compression_evidence_20261024/README.md) · [causal LM](causal_lm_evidence_20261024/README.md) | Train-separated quality controls |

## 4. Certified co-training groups and bag queries

| Report | Result and boundary |
| --- | --- |
| [Matched co-training and planner controls](cotraining_evidence_20261031/README.md) · [claims](cotraining_contribution_claims.md) ([ledger](cotraining_claims_20261101/)) | 1.86M-row subset groups to 360k parameter rows; raw certified queries 33.5× faster than encoded but trail raw mmap |
| [FAISS PQ/OPQ/RQ/clustering controls](faiss_cotraining_evidence_20261101/README.md) · [protocol](faiss_cotraining_protocol.md) | Same-task codec comparison and fixed-budget capacity |
| [Frozen FAISS query schedule](faiss_schedule_evidence_20261102/README.md) · [protocol](faiss_schedule_protocol_20261102.md) | Batch compilation with identical outputs |
| [Independent corpus and fused native query](cross_corpus_evidence_20261102/README.md) · [protocol](cross_corpus_protocol_20261102.md) · [fused plan](native_fused_plan_protocol_20261102.md) | enwik8: 12.33–40.47% extra saving at 0.1% loss budget; native total path 3.04× faster with identical bytes |
| [Disk-backed incidence construction](external_incidence_evidence_20261102/README.md) · [protocol](external_incidence_protocol_20261102.md) | Build RSS 3,140.7 → 95.0 MiB (33.06×), time 23.3 → 116.6 s; identical bytes |
| [Certified native build and PyTorch CPU](certified_serving_evidence_20261103/README.md) · [protocol](certified_serving_protocol_20261103.md) | Build RSS 3,143.5 → 100.7 MiB; three-file bundle; original-ID queries match |
| [Partial-query error budgets](budgeted_bag_evidence_20261104/README.md) · [protocol](budgeted_bag_protocol_20261104.md) | 1% L2 bounds admit 98.8–99.8% of OOV bags vs 9.6–16.2% strict coverage |
| [Native bounded-plan compiler](native_bounded_query_evidence_20261105/README.md) · [protocol](native_bounded_query_protocol_20261105.md) | 1.77–2.42× faster assessment with identical outputs; source FP64 still faster |

## 5. Applications and LLM integration

| Report | Result and boundary |
| --- | --- |
| [AG News CPU application](news_classifier_evidence_20261105/README.md) · [protocol](news_classifier_protocol_20261105.md) · [ReLU control](news_relu_protocol_20261105.md) | 612,759 rows, 7,600 test docs × 3 seeds: INT8 bundles 177.66 → 45.14 MiB, identical predictions |
| [Independent news / hybrids](news_shift_evidence_20261105/README.md) · [protocol](news_shift_protocol_20261105.md) | Native INT8 hybrid 37.84 vs 60.83 MiB; PQ/precomputed alternatives win qualified budgets |
| [Actual FastText model](fasttext_native_evidence_20261031/README.md) | 90,380 rows byte-identical; bundle saves 0.46%; hot queries slower than dense |
| [TinyEngram/Qwen checkpoint](tinyengram_qwen_evidence_20261030/README.md) · [CUDA timing](qwen_cuda_performance_20261030/README.md) · [A100 correctness](cuda_a100_evidence_20261030/README.md) · [A100 gather](cuda_gather_evidence_20261030/README.md) | Trained outputs preserved; native lookup slower than dense offload on these traces |
| [Device memory and prefetch](device_memory_evidence_20261030/README.md) · [device gather baselines](device_gather_benchmark.md) ([prep](device_gather_preparation_20261030/README.md)) | Host-to-device gather costs |
| Official Engram demo: [gather interface](engram_adapter_verification_20261018.md) · [full forward](engram_full_adapter_verification_20261018.md) · [row-store smoke](engram_rowstore_smoke_20261018.md) | Byte-exact storage replacement on the small official demo (not a trained Engram checkpoint) |

Supporting JSON without a separate README: `controlled_*_results_20261111.json`,
`gpu_pair_block_scheduling_addendum_20261111.json`, `row_packet_protocol_20261111.json`,
`training_participation_sensitivity_20261111.json`, `qwen_*_protocol_20261106.json`,
`engram_*_20261018/19.json`.
