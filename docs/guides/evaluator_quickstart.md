# Engine guide: pack, inspect, query

This guide packs a float32 table into a native store and queries it from the CLI,
Python and C. Terms are explained in [concepts](../concepts.md).

## Install

You need a C compiler and zlib headers, on Linux or macOS.

```bash
python -m pip install .
ngram-store --help
```

To run from a checkout without installing, build the library and point to it:

```bash
make native                                            # Linux
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib  # macOS
export MEFT_ROWS_LIBRARY=$PWD/build/libmeft_rows.so     # .dylib on macOS
python -m ngram_store --help
```

The pip distribution is still called `meft-memory`; import `ngram_store`. Old
`meft_memory` imports and `meft-*` commands point to the same code.

## Choose a layout

```bash
ngram-store plan --input vectors.f32 --dim 256
```

`plan` hashes every row, counts duplicates, and prints the exact size of two
layouts whose lookups only copy rows: raw, and deduplicated. It recommends the
smaller one and prints the `pack` command for it. Add `--prefer size` to also
estimate block-compressed layouts from a sample; those are smaller when rows
compress, but every lookup decodes whole blocks and can be hundreds of times
slower on scattered IDs.

On the twelve AG News/DBpedia classifier tables of the controlled grid, the
predicted uncompressed sizes matched the packed files to the byte, `--prefer size`
picked the same codec family as the grid's calibration and estimated its size
within 0.15%, and the default picked deduplication for zero-initialized tables
and raw for uniform ones ([validation](../../evidence/host_query_profile_20261112/README.md#layout-planner)).

## Pack a table

Input is a headerless, row-major, little-endian float32 file. For a quick test,
make one with 1,000 rows of width 8, a third of them zero:

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, (1000, 8)).astype("<f4")
x[::3] = 0
x.tofile("vectors.f32")
```

```bash
ngram-store pack --input vectors.f32 --dim 8 --output memory.native
```

`pack` reads the table back from the new store and compares every row with the
source before publishing the directory. The JSON it prints includes:

```text
"original_bytes_preserved": true,
"input_bytes": 32000,
"artifact_bytes": 20670,
"forwarding": "implicit-identity",
"blocks": 4,
```

The output directory must not exist yet. Useful options:

| Option | Effect |
| --- | --- |
| `--max-block-bytes N` | Cap decoded block size (default 8192). Smaller blocks speed up sparse lookups and compress less |
| `--deduplicate` | Store identical rows once, at 8 bytes per row for the row map |
| `--no-compress` | Store blocks uncompressed. With `--deduplicate`, lookups copy rows without decoding |

Deduplication does not always pay. On the table above, `--deduplicate` keeps
667 unique rows but the file grows from 20,670 to 27,931 bytes, because the
8,000-byte row map costs more than the duplicate rows saved.

Compression costs query time. Every lookup decodes a whole block, and the IDs of
one query are often spread over many blocks. On a trained 7M-row AG News table,
`--deduplicate --no-compress` saved 85.8% against 86.6% for dedup + Zstd, while
lookups ran 1.2× raw mmap instead of 356×
([profile](../../evidence/host_query_profile_20261112/README.md)). If the table has
many duplicate rows, start with `--deduplicate --no-compress`; if it has few, keep
it raw.

## Inspect and query

```bash
ngram-store inspect --store memory.native
ngram-store query --store memory.native --ids 1 0 1
ngram-store query --store memory.native --ids 0 --hex
```

`query` returns rows in request order, repeats included. `--hex` prints the raw
bytes, which also works for NaN and infinity (the JSON output needs finite
values). `--cache-bytes` sets the decoded-block cache for that call.

From Python:

```python
import numpy as np
from ngram_store import NativeRows

with NativeRows("memory.native/table.men", cache_bytes=4 * 1024**2) as table:
    raw = table.rows([1, 0, 1])          # bytes, 3 × 32
    vectors = table.gather_array(np.array([1, 0, 1])).view("<f4")   # (3, 8)
```

`rows` takes Python integers; `gather_array` takes a NumPy integer array and skips
the per-ID Python work, which matters when rows need no decoding.

From C, see `examples/native_gather.c` and the
[C ABI reference](native_ngram_memory.md):

```bash
build/native_gather memory.native/table.men 1 0 1
```

## String keys (Linux x86-64)

If each row has a unique string key, such as a character n-gram, the structural
layer can index the keys with a DATrie and share prefix/suffix rows:

```bash
make structural-native
ngram-store pack --input vectors.f32 --dim 256 --output keyed.native \
  --keys keys.json --structural-library build/libmeft_structural.so --mode exact
ngram-store query --store keyed.native --keys '<memory>' '<vector>' \
  --structural-library build/libmeft_structural.so
```

`keys.json` maps row IDs to strings, e.g. `{"0": "<memory>", "42": "<vector>"}`;
rows without a key keep their IDs. Modes:

- `layout`: reorder rows only.
- `exact`: also share byte-identical prefix/suffix rows.
- `joint`: also merge rows whose cosine and norm ratio pass `--cosine` and
  `--norm-ratio` (default .99). This changes vectors.

Hash-addressed tables (Engram, FastText buckets) should use plain `pack`: a bucket
can hold unrelated n-grams, so its string relations mean nothing. Only open DATrie
files you produced yourself.

## Next

- Plug the store into a model: [FastText](fasttext_native_bundle.md),
  [Engram](engram_quickstart.md),
  [TinyEngram/Qwen](../../evidence/tinyengram_qwen_evidence_20261030/README.md),
  [PyTorch adapters](native_ngram_memory.md#pytorch).
- Compress bag-reduced tables further: [certified bag tables](try_certified_memory.md).

## Reporting a benchmark

Results are comparable when they include the commit, data and split, source and
artifact hashes, hardware and software versions, and the same trace run against
raw mmap or dense lookup. Report total artifact bytes, p50/p95 latency, load time,
RSS, cache and block settings, output fidelity and task quality as separate
numbers, and keep slow or failed runs. The reports in
[`evidence/`](../../evidence/README.md) follow this shape.
