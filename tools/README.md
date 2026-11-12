# tools/

This directory holds two kinds of Python files:

- **Runtime modules** (22). The installed package imports these. In a wheel they
  appear as `meft_memory._runtime.<name>`, and `ngram_store` re-exports their
  public classes.
- **Experiment scripts** (163). Benchmarks, training runs, analyses and audits
  that produced the reports in [`evidence/`](../evidence/README.md).

Both stay at these paths: evidence manifests record their hashes, and verifiers
import them as `tools.<name>`.

## Runtime modules

| Module | Purpose |
| --- | --- |
| `native_row_store` | ctypes bindings for the C library; `NativeRows`, `pack_native` |
| `block_compression` | Optional system zlib/LZ4/Zstd block compressors |
| `lossless_row_store` | Pure-Python reference store (exact-byte dedup + block lookup) |
| `quantized_memory` | FP16 / INT8 row codecs on top of native storage |
| `faiss_row_codec` | FAISS codecs used for original-ID lookup (not ANN search) |
| `structural_native` | DATrie prefix/suffix planner and mark-compact export |
| `engram_row_adapter` | PyTorch `MappedEmbedding`, `MappedMultiHeadEmbedding` and exporters |
| `mapped_embedding_bag` | PyTorch `MappedEmbeddingBag` over native storage |
| `device_row_adapter` | Host-resident memory with bounded staging and CUDA prefetch |
| `cotraining_groups` | Certified incidence groups, `CertifiedBagRows` |
| `external_cotraining` | Disk-backed incidence construction and group projection |
| `certified_store` | `compact-bags --native`: build a ready-to-query certified bundle |
| `certified_torch` | PyTorch `CertifiedEmbeddingBag` |
| `budgeted_bags` | L2-bounded partial-group queries |
| `demo_certified_store` | `demo-bags`: offline SGNS training → certified bundle |
| `text_classifier` | Text → n-gram bags → linear classifier, NumPy/C serving |
| `prepare_fasttext_corpus` | Plain text → input for the custom FastText trainer |
| `fasttext_native_bundle` | Export a me_fasttext `.bin` to a native serving bundle |
| `tinyengram_checkpoint` | TinyEngram/Qwen3 checkpoints with native memory |
| `train_tinyengram_reference` | Train a small Engram branch on a local Qwen3 |
| `verify_tinyengram_checkpoint` | Replay a checkpoint + native bundle on CPU |
| `verify_tinyengram_device` | Compare resident vs staged device memory |

`certified_sample.txt` is the demo corpus; `fasttext_serving_probe.cc` is a C++
probe used by the FastText benchmark.

## Experiment scripts by prefix

| Prefix | Count | Role |
| --- | --- | --- |
| `benchmark_*` | 45 | Run a measurement and write a raw report |
| `analyze_*` | 18 | Turn raw reports into published summaries |
| `counterfact_*` | 11 | MultiCounterFact data, training and evaluation |
| `plot_*` | 9 | Figures in evidence READMEs |
| `train_*` | 8 | Training runs for SGNS, classifiers and TinyEngram |
| `audit_*` | 7 | Cross-checks on completed grids |
| `check_*` | 5 | CI checks: wheels, sanitizers, legacy roundtrip |
| `demo_*` | 5 | Small demonstrations (Engram adapter, lossless rows) |
| `verify_*` | 5 | Replay or re-verify specific results |
| `run_*`, `gpu_*`, `finalize_*`, `package_*`, … | ~50 | Scheduling, resumption and packaging of large grids |

To find the script behind a report, look at the commands listed in that report's
README or `manifest.json`.
