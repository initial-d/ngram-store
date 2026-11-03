# Try a trained n-gram memory table in minutes

NGramStore can construct and serve a frozen, certified n-gram embedding table
using NumPy and its standalone native C reader. Start with the included offline
example; then use your own existing FP32 table and original-ID bags.

## Install v0.4.0

Python 3.10+; Linux x86-64 or macOS ARM64. The distribution remains
`meft-memory` to preserve historical installs. Use the matching native wheel:

```bash
# Linux x86-64
python -m pip install 'meft-memory[incidence] @ https://github.com/initial-d/ngram-store/releases/download/v0.4.0/meft_memory-0.4.0-py3-none-linux_x86_64.whl'

# macOS ARM64 (macOS 26+)
python -m pip install 'meft-memory[incidence] @ https://github.com/initial-d/ngram-store/releases/download/v0.4.0/meft_memory-0.4.0-py3-none-macosx_26_0_arm64.whl'
```

Source builds can use `python -m pip install '.[incidence]'` from a checkout,
with a C compiler and zlib. Release wheels are on GitHub; this guide does not
assume that PyPI has this version. Installed wheels include the example corpus.

## Train, compact and query the offline example

```bash
ngram-store demo-bags --output demo
ngram-store inspect --store demo/memory
ngram-store query-bags --store demo/memory --bags demo/queries.json --mode mean
```

The example performs actual CPU SGNS updates on a small MIT text authored in
this repository, then builds a native table, checks every canonical row,
evaluates certified aggregates and rejects an incomplete group. It needs no
external dataset, model download, Torch or GPU. Its report describes training
and rounding; it is an integration example, not a capacity/quality benchmark.

The release also includes the standalone
[reproduction script](../examples/reproduce_certified_demo.py). Run it outside
the checkout with the installed wheel:

```bash
python reproduce_certified_demo.py --output my-reproduction
```

It independently invokes the installed CLI, compares against the legacy
partition/projection and checks that incomplete queries are refused. Running
this yourself is a reproducible entry; third-party adoption is a separate outcome.

## Use your own trained table

```bash
ngram-store compact-bags --native \
  --input vectors.f32 --dim 256 \
  --bag-ids ids.u64 --bag-offsets offsets.u64 \
  --protected-ids protected.u64 \
  --memory-mib 64 --output memory

ngram-store query-bags --store memory --ids 0 1 0 1 --mode mean
```

`ids.u64` contains the complete original-ID occurrences of each registered bag,
including duplicates. `offsets.u64` starts at zero and ends at the ID count;
retain empty bags. Protect independently accessed word/padding IDs, or omit
that argument when none apply. The serving directory contains only
`table.men`, `domain.npz` and `manifest.json`; expanded canonical rows and
intermediate means are removed. Original IDs forward to the group means.

Python queries use the same certificate:

```python
from ngram_store import CertifiedBagRows

with CertifiedBagRows("memory") as memory:
    vectors = memory.aggregate([[0, 1, 0, 1], []], mean=True)
```

Queries are accepted only when each used group's members have equal counts.
The sample IDs above are illustrative: your table determines which queries
are certified. `query-bags` refuses a batch containing an uncertified query;
ordinary `query` refuses certified artifacts to prevent accidental raw-row use.

## PyTorch CPU integration

With the optional Torch dependency installed:

```python
import torch
from ngram_store import CertifiedEmbeddingBag

with CertifiedEmbeddingBag("memory", mode="mean", include_last_offset=True) as lookup:
    vectors = lookup(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 4]))
```

The module supports CPU int32/int64 fixed or ragged bags, sum/mean, optional
padding and frozen inference. It owns no trainable parameters. Max pooling,
per-sample weights, CUDA tensors and arbitrary uncertified bags require
different contracts. It is a package integration with PyTorch, not an upstream
PyTorch endorsement or merge. The standalone runner's `--torch` option also
checks a trained example SGNS output head.

Projection changes individual source rows. Certified sums are preserved in real
arithmetic, with FP32 rounding measured separately; native FP64 reduction is
not automatically byte-identical to PyTorch's FP32 reduction. Keep an original
dense option for queries outside the supported domain.

The configured memory budget manages build buffers; it does not cap RSS or OS
page cache. Corpus preparation, runtime map loading and query compilation are
separate costs. See [construction evidence](external_incidence_evidence_20261102/README.md)
and the [complete serving protocol](certified_serving_protocol_20261103.md).
Engram's independent fields/gates still need model-specific validation before
using this aggregate transformation. Ordinary original-row byte-preserving
storage remains available through `pack/query`.
