# History: me_fasttext

NGramStore began as **me_fasttext**, a FastText variant prototyped in 2017. It
replaced hashed subword buckets with exact double-array-trie (DATrie) identities,
trained one row per retained n-gram, then shared and compacted rows after
training in a mark-compact style before writing an mmap-ready `.z` artifact. The
idea was written up in

> Yimin Du. *Memory-Efficient FastText: A Comprehensive Approach Using
> Double-Array Trie Structures and Mark-Compact Memory Management.*
> arXiv:2506.01254.

NGramStore keeps the idea (exact identity first, compaction later) and rebuilt
everything else: the storage format, codecs, query paths and LLM adapters. Its
measurements are in the [evidence archive](../../evidence/README.md).

The originally released exporter merged rows on a norm-ratio test alone; the
current one also checks cosine similarity and prefix/suffix structure
([audit](../guides/engram_storage.md#legacy-export-audit),
[roundtrip check](../../evidence/audits/legacy-roundtrip-20261024/README.md)).

## Documents from that period

**Design and retrospectives**

- [技术回顾：把动态内存管理用在 FastText 的 n-gram embedding 上](technical_retrospective_zh.md)
- [Engineering note: exact subword ids, compact rows, mmap serving](engineering_note.md)
- [Memory management for AI embeddings](memory_management_for_ai_embeddings.md)
- [Early compact-inference ideas](inference_optimization_note.md)

**Positioning (lexical retrieval / conditional memory)**

- [Related work map](related_work_map.md)
- [Indexed lexical memory and conditional memory](conditional_memory_context.md)
- [Model-compression and Engram bridge](model_compression_engram_bridge.md)
- [Artifact card](artifact_card.md)

**Early benchmark material**

- [Benchmark protocol](benchmark_protocol.md) · [first public-corpus benchmark](first_public_corpus_benchmark.md)
- [Retrieval / OOV serving benchmark](retrieval_oov_benchmark.md) · [retrieval comparison matrix](retrieval_comparison_matrix.md)
- [Compression comparison checklist](compression_comparison_checklist.md)
- [v0.1.0 release notes](release_notes_v0.1.0.md)

Compatibility with old imports, commands and file formats is maintained; see
[legacy_fasttext.md](../guides/legacy_fasttext.md).
