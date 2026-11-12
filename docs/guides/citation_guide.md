# Citation guide

## Cite NGramStore

Cite the software with the exact commit, and the evidence report if you rely on
a specific measurement. GitHub's "Cite this repository" button reads
[`CITATION.cff`](../../CITATION.cff).

```bibtex
@software{du_ngramstore,
  author = {Yimin Du},
  title  = {NGramStore: Compression, Storage and Exact Lookup for Learned N-Gram Embeddings},
  url    = {https://github.com/initial-d/ngram-store},
  note   = {Commit [full SHA]; evidence report [evidence/<dir> and date]}
}
```

Results belong to the specific report and commit they were produced with; see
the [evidence archive](../../evidence/README.md). When quoting a number, keep the
boundary stated in that report (trained table, corpus, baseline, lossless vs
approximate, byte denominator).

## Origin of the idea

The exact-identity-then-compact idea dates back to the 2017 me_fasttext prototype,
written up as:

```bibtex
@misc{du2025memoryefficientfasttext,
  title         = {Memory-Efficient FastText: A Comprehensive Approach Using Double-Array Trie Structures and Mark-Compact Memory Management},
  author        = {Yimin Du},
  year          = {2025},
  eprint        = {2506.01254},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2506.01254}
}
```

Cite it for the original idea. NGramStore's engine, adapters and measurements
are later work; please attribute them to the software and evidence reports, not
to the paper. See [history](../history/README.md).

## Suggested wording

For the engine:

> NGramStore stores frozen learned n-gram tables in compressed native blocks and
> serves original row IDs through mmap, decoded caches and batched gathers,
> preserving row bytes in its lossless modes.

For Engram-related use, keep the tested boundary:

> Small source-pinned Engram demo checks preserve gated module output; separate
> TinyEngram/Qwen checks do not establish trained DeepSeek Engram or production
> serving results.
