# NGramStore

Compression, storage and exact lookup for learned n-gram embedding tables.

面向 learned n-gram embedding 表的压缩、存储与精确查询。

[![CI](https://github.com/initial-d/ngram-store/actions/workflows/ci.yml/badge.svg)](https://github.com/initial-d/ngram-store/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/initial-d/ngram-store?label=release)](https://github.com/initial-d/ngram-store/releases/latest)

FastText builds word vectors from subword n-gram rows. Engram-style LLM memory
looks up token n-gram rows and mixes them into hidden states. In both cases the
model ends up owning a large frozen table that has to be indexed, stored and
queried. NGramStore takes care of that table; training, addressing and
aggregation stay in the model.

The project also asks research questions: when is it worth giving every n-gram
its own row instead of hashing into buckets, what redundancy does training leave
in the table, and how should it be served? The
[technical report](docs/report/ngramstore-report.pdf) answers them; the
experiments behind it are in [`evidence/`](evidence/README.md).

## Components

| Layer | What it does |
| --- | --- |
| Table engine | Lossless block encoding, original-ID forwarding, mmap, decoded-block cache, batched gather. C ABI and Python |
| N-gram structure (optional) | DATrie index over registered string keys, prefix/suffix representative selection, mark-compact export |
| Model adapters | FastText, Engram / TinyEngram / Qwen memory, CPU text classifiers |

`pack` and the structural `layout`/`exact` modes keep source bytes. `joint`
sharing (cosine + norm-ratio guard) and quantization change vectors and are
opt-in.

## Quick start

Needs a C compiler and zlib headers, on Linux or macOS. PyTorch is optional.

```bash
python -m pip install .
```

Pack an existing table (row-major little-endian float32) and query it:

```bash
ngram-store pack    --input vectors.f32 --dim 256 --output memory.native
ngram-store inspect --store memory.native
ngram-store query   --store memory.native --ids 0 42 0
```

```python
from ngram_store import NativeRows

with NativeRows("memory.native/table.men", cache_bytes=4 * 1024**2) as memory:
    original_bytes = memory.rows([0, 42, 0])
```

`ngram-store plan --input vectors.f32 --dim 256` counts duplicate rows and
prints the `pack` command for the smallest layout that still serves lookups at
near raw speed.

`ngram-store demo-bags --output demo` trains a small model on bundled text and
builds a [certified bag table](docs/guides/try_certified_memory.md) end to end.
The [engine guide](docs/guides/evaluator_quickstart.md) walks through a complete
example, and [concepts](docs/concepts.md) explains the terms.

## Results

1. Distinct rows vs hash buckets. On MultiCounterFact with 20,877 edits on frozen
   Qwen 0.6B/4B, independent n-gram rows improve H by 0.41/1.24 pp over hashing
   with matched coverage ([report](evidence/ngram_identity_evidence_20261107/README.md)).
   On smaller grids, held-out perplexity and joint training the gap disappears
   ([details](evidence/README.md#1-does-exact-n-gram-identity-beat-hash-buckets)).
2. Prefix/suffix structure is learned. Merging order-12 prefix/suffix neighbours
   at .99 cosine saves 1.34% of n-gram storage after 2k training steps and 41.49%
   after 32k ([report](evidence/tree_node_quality_evidence_20261109/README.md)).
3. Duplicate rows, not compression, carry the lossless saving. On zero-initialized
   classifier tables, byte dedup + Zstd saves about 84% but made host lookups
   29–79× slower in the deployment grids
   ([report](evidence/controlled_deployment_results_20261112.md)). Profiling
   showed the cost is block decompression. Dedup without compression saves
   85.0–85.8% (vs 85.7–86.6% with Zstd), and the same GPU host path runs
   1.09–1.23× raw with identical logits
   ([profile](evidence/host_query_profile_20261112/README.md)). Without
   duplicates (uniform initialization) compression saves only 8–9%; raw is better.
4. Construction fits in small memory. Disk-backed building cuts peak RSS from
   3.1 GiB to 95 MiB for the same output bytes, at 5× the build time
   ([report](evidence/external_incidence_evidence_20261102/README.md)).

All ~45 reports, grouped by question, are listed in the [evidence index](evidence/README.md).

## Applications

| Application | Guide |
| --- | --- |
| FastText word embeddings | [prepare → train → export → query](docs/guides/fasttext_native_bundle.md) |
| Engram LLM memory | [official demo and factory](docs/guides/engram_quickstart.md) |
| TinyEngram / Qwen | [checkpoint export and decode](evidence/tinyengram_qwen_evidence_20261030/README.md) |
| CPU news classification | [AG News](docs/guides/news_classifier_app.md) · [open-vocabulary hybrids](docs/guides/news_shift_app.md) |

## Repository layout

| Path | Contents |
| --- | --- |
| `native/` | C engine (`meft_rows.c`) and structural bridge |
| `ngram_store/`, `meft_memory/` | Python package and CLI |
| [`tools/`](tools/README.md) | 22 runtime modules and the experiment scripts behind `evidence/` |
| `src/`, `mylib/` | FastText-derived C++ trainer/exporter and DATrie libraries (Linux x86-64) |
| `tests/`, `examples/`, `benchmarks/` | Tests, runnable examples, benchmark configs |
| [`docs/`](docs/index.md) | Guides, roadmap, history |
| [`evidence/`](evidence/README.md) | Experiment protocols, results and verifiers |

The C engine and wheel run on Linux and macOS; the DATrie bridge and FastText
facade need Linux x86-64. Tables are read-only at serving time. Runtime details
are in the [C ABI guide](docs/guides/native_ngram_memory.md); what comes next is
in the [roadmap](docs/roadmap.md).

## History

NGramStore started as me_fasttext, a 2017 FastText prototype that gave every
n-gram an exact DATrie ID and compacted rows after training, in the spirit of a
mark-compact collector ([arXiv:2506.01254](https://arxiv.org/abs/2506.01254)).
The engine has since been rebuilt. Old `meft_memory` imports, `meft-*` commands
and file formats still work ([compatibility](docs/guides/legacy_fasttext.md),
[history](docs/history/README.md)).

## Citing and contributing

Cite the software through [`CITATION.cff`](CITATION.cff) with the commit you used
([citation guide](docs/guides/citation_guide.md)). Benchmark results and
integration reports are welcome; see [CONTRIBUTING](CONTRIBUTING.md).

## License

MIT. Bundled FastText sources keep their original notices. See [LICENSE](LICENSE).
