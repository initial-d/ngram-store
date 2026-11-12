# Certified bag tables

Many n-gram models only ever read their table through sums or means over bags of
IDs: FastText averages a word's n-grams, a classifier pools a document. For those
models, rows that always appear together in the same bags can be stored as one
shared mean, and every complete bag sum stays exact. This guide builds such a
table, queries it, and plugs it into PyTorch. The background is in
[concepts](../concepts.md#bags-and-certified-groups) and the math in
[certified co-training groups](cotraining_groups.md).

Individual rows change in this mode. If you need the original rows back, use plain
[`pack`](evaluator_quickstart.md) instead.

## Install

Python 3.10+, NumPy, Linux x86-64 or macOS ARM64. Release wheels:

```bash
# Linux x86-64
python -m pip install 'meft-memory[incidence] @ https://github.com/initial-d/ngram-store/releases/download/v0.4.0/meft_memory-0.4.0-py3-none-linux_x86_64.whl'

# macOS ARM64 (macOS 26+)
python -m pip install 'meft-memory[incidence] @ https://github.com/initial-d/ngram-store/releases/download/v0.4.0/meft_memory-0.4.0-py3-none-macosx_26_0_arm64.whl'
```

Or from a checkout: `python -m pip install '.[incidence]'` (needs a C compiler and
zlib). Wheels are on GitHub releases, not PyPI.

## Run the included example

```bash
ngram-store demo-bags --output demo
ngram-store inspect --store demo/memory
ngram-store query-bags --store demo/memory --bags demo/queries.json --mode mean
```

`demo-bags` trains a small skip-gram model on a short text bundled with the
package (CPU, NumPy, about a thousand rows), builds the certified native table,
checks every group, and shows that a bag covering only part of a group is
rejected. No downloads, PyTorch or GPU needed.

[`examples/reproduce_certified_demo.py`](../../examples/reproduce_certified_demo.py)
does the same through the installed CLI from outside the checkout and compares the
result with the older pure-Python path:

```bash
python reproduce_certified_demo.py --output my-reproduction
```

## Use your own table

You need three raw files:

- `vectors.f32`: the trained table, little-endian float32, row-major.
- `ids.u64`: every bag's original IDs, concatenated, duplicates included.
- `offsets.u64`: bag start positions, beginning with 0 and ending with the total
  ID count. Keep empty bags.

Optionally, `protected.u64` lists IDs that are also read on their own (whole words,
padding); they stay unshared.

```bash
ngram-store compact-bags --native \
  --input vectors.f32 --dim 256 \
  --bag-ids ids.u64 --bag-offsets offsets.u64 \
  --protected-ids protected.u64 \
  --memory-mib 64 --output memory

ngram-store query-bags --store memory --ids 0 1 0 1 --mode mean
```

The build streams through disk with a 64 MiB buffer budget. The result has three
files: `table.men`, `domain.npz` and `manifest.json`. On a 1.86M-row table this
cut peak build RSS from 3,143.5 to 100.7 MiB and took 4.1× longer
([report](../../evidence/certified_serving_evidence_20261103/README.md)). The
buffer budget is not a hard RSS limit; the OS page cache and map loading come on
top.

Which bags are accepted depends on your table: a bag passes when, for every group
it touches, each member appears the same number of times. `query-bags` refuses a
batch containing any other bag, and plain `query` refuses certified stores so they
cannot be read as ordinary rows by mistake. To accept some of those bags within a
measured error, see [partial-group queries](budgeted_bag_queries.md).

## Python and PyTorch

```python
from ngram_store import CertifiedBagRows

with CertifiedBagRows("memory") as memory:
    vectors = memory.aggregate([[0, 1, 0, 1], []], mean=True)
```

With PyTorch installed, `CertifiedEmbeddingBag` is a drop-in for inference with
`nn.EmbeddingBag`:

```python
import torch
from ngram_store import CertifiedEmbeddingBag

with CertifiedEmbeddingBag("memory", mode="mean", include_last_offset=True) as lookup:
    vectors = lookup(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 4]))
```

It supports CPU int32/int64 IDs, fixed or ragged bags, sum and mean, and padding.
It has no trainable parameters, and does not support max pooling, per-sample
weights or CUDA tensors.

## Precision

Group means keep complete-group sums exact in real arithmetic. In floating point
the native reducer accumulates in FP64 while PyTorch's FP32 reduction rounds
differently, so the last bits can differ from the source model. Keep the original
table available if you need to serve bags outside the certified set.

Engram-style memory with separate per-field gates needs its own validation before
using this transformation.
