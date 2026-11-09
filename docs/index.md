# NGramStore guides and evidence

## Engine and application entry points

- [Try a trained certified memory table: release wheel, offline demo and own-table flow](try_certified_memory.md)
- [Engine: pack, inspect and query](evaluator_quickstart.md)
- [C ABI, native construction and runtime boundaries](native_ngram_memory.md)
- [Engine roadmap, priorities and acceptance gates](ngram_memory_research_plan.md)
- [Optional certified co-training parameterization and export API](cotraining_groups.md)
- [Opt-in partial-group queries with source-specific L2 budgets](budgeted_bag_queries.md)
- [Disk-backed incidence construction: NumPy-only CLI and buffer contract](external_incidence_build.md)
- [Co-training paper contributions, evidence and open hypotheses](cotraining_contribution_claims.md)
- [Word embeddings: FastText prepare/train/export/query](fasttext_native_bundle.md)
- [CPU news application and independent precomputed/subword hybrids](news_shift_app.md)
- [LLM memory: official Engram demo and factory](engram_quickstart.md)
- [TinyEngram/Qwen checkpoint integration](tinyengram_qwen_evidence_20261030/README.md)
- [Historical me_fasttext compatibility](legacy_fasttext.md)
- [Citation guide](citation_guide.md) and [artifact card](artifact_card.md)

## Source-linked trained results

- [Real Qwen native offload: complete logit pairs and storage costs](qwen_native_identity_evidence_20261109/README.md)
- [Identity format: remove redundant maps on the same trained Qwen tables](identity_format_evidence_20261109/README.md)
- [Independent n-gram keys: complete 0.6B/4B matched hash controls and paired uncertainty](ngram_identity_evidence_20261107/README.md)
- [Eight-GPU Qwen: trained identity controls, complete codecs and faster native hot paths](qwen_gpu_grid_evidence_20261106/README.md)
- [Held-out Qwen: memory branch effect, head-constant control and native output identity](qwen_quality_evidence_20261106/README.md)
- [Independent news: 15 controls, fair hybrids and the necessity boundary](news_shift_evidence_20261105/README.md)
- [Actual legacy FastText model and matched layout controls](fasttext_native_evidence_20261031/README.md)
- [Original DATrie / mark-compact contribution ablations](structural_native_evidence_20261030/README.md)
- [Same-trained-table FP16/INT8 and structural-sharing controls](quantized_structural_evidence_20261031/README.md)
- [1.85M trained-row CPU capacity](cpu_capacity_evidence_20261030/README.md)
- [Large FP16/INT8 capacity, construction RSS and fresh CPU traces](quantized_capacity_evidence_20261031/README.md)
- [Large-table structural applicability, shared updates and initialization](large_structural_evidence_20261031/README.md)
- [Matched co-training, 1.86M-row capacity and independent DATrie controls](cotraining_evidence_20261031/README.md)
- [FAISS PQ/OPQ/RQ/clustering controls, fixed-budget capacity and batch certification](faiss_cotraining_evidence_20261101/README.md)
- [Frozen FAISS codec query: batch compilation, identical outputs and total latency](faiss_schedule_evidence_20261102/README.md)
- [Independent corpus: matched training, complete codec controls and fused native total query](cross_corpus_evidence_20261102/README.md)
- [External incidence construction: full byte comparisons, RAM/time/disk and native compatibility](external_incidence_evidence_20261102/README.md)
- [Complete certified native serving: build resources, installed wheels and actual PyTorch CPU trace](certified_serving_evidence_20261103/README.md)
- [Actual-source error budgets: partial/OOV coverage, complete capacity and query costs](budgeted_bag_evidence_20261104/README.md)
- [Sparse-block native miss latency](sparse_blocks_evidence_20261030/README.md)
- [Native query/cache controls](native_query_evidence_20261030/README.md)
- [Causal language-model quality controls](causal_lm_evidence_20261024/README.md)
- [Held-out lossless compression](heldout_compression_evidence_20261024/README.md)
- [Native embedding-bag memory and latency](embedding_bag_evidence_20261025/README.md)
- [Trained n-gram similarity](trained_ngram_similarity_20261018.md)
- [Approximate vector distortion curves](approximate_ngram_compression_20261018.md)
- [Qwen CUDA inference timing controls](qwen_cuda_performance_20261030/README.md)
- [A100 output correctness](cuda_a100_evidence_20261030/README.md)

These reports keep their original run dates, source commits, commands and
historical project names. A rename does not change their evidence or scope.
Official Engram demo fixtures, trained references and TinyEngram/Qwen results
are separate categories. Raw mmap/dense controls and no-benefit results remain
close to claimed savings. [Contribution/reporting guide](../CONTRIBUTING.md).
