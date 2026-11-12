# C ABI and runtime reference

The native backend serves immutable embedding tables on CPU. It takes original
numeric row IDs and writes the stored bytes into memory the caller owns. The
model keeps its tokenizer, hashing, gates and training. Linux and macOS are
supported; Windows, CUDA kernels and distributed serving are not.

- [Build](#build)
- [C API](#c-api)
- [Native layout and construction](#native-layout-and-construction)
- [Blocks and cache](#blocks-and-cache)
- [PyTorch](#pytorch)
- [Embedding bags](#embedding-bags)
- [Quantized rows](#quantized-rows)
- [Model integration](#model-integration)
- [Tests](#tests)

## Build

```bash
make native                                             # Linux
make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib  # macOS
```

This builds `build/libmeft_rows.{so,dylib}` and the example `build/native_gather`
from [`native/meft_rows.c`](../../native/meft_rows.c). It needs only zlib and POSIX
mmap, not the bundled FastText/DATrie libraries.

Optional codecs, if the development libraries are installed:

| Flag in `NATIVE_CODEC_FLAGS` | Wheel env var | Effect |
| --- | --- | --- |
| `-DMEFT_USE_LIBDEFLATE -ldeflate` | `MEFT_USE_LIBDEFLATE=1` | Faster decoding of existing zlib blocks; files unchanged |
| `-DMEFT_USE_LZ4 -llz4` | `MEFT_USE_LZ4=1` | Read and write `codec="lz4"` files |
| `-DMEFT_USE_ZSTD -lzstd` | `MEFT_USE_ZSTD=1` | Read and write `codec="zstd"` files |

A reader built without a codec fails when opening a file that uses it.
`NativeRows.decoder_name` reports which zlib implementation is in use.

## C API

[`native/meft_rows.h`](../../native/meft_rows.h) is a plain C ABI usable from C,
C++ and FFI. (The older `src/ftindex.h` uses `std::string` and `std::vector`, so it
is C++ only.)

```c
meft_rows *meft_rows_open(const char *path, char *error, size_t error_capacity);
void       meft_rows_close(meft_rows *store);
uint64_t   meft_rows_count(const meft_rows *store);   /* rows */
uint64_t   meft_rows_width(const meft_rows *store);   /* bytes per row */

int meft_rows_gather(meft_rows *store, const uint64_t *ids, size_t count,
                     void *output, size_t output_bytes,
                     char *error, size_t error_capacity);
int meft_rows_gather_chunked(..., size_t max_query_rows, ...);

int meft_rows_set_cache_bytes(meft_rows *store, size_t budget);
int meft_rows_set_cache_policy(meft_rows *store, int rows);
```

- `output` must hold `count × width` bytes.
- `meft_rows_gather_chunked` sorts at most `max_query_rows` request locations at a
  time, bounding that scratch memory. Chunk boundaries can decode a block twice.
- `meft_rows_embedding_bag` and `meft_rows_embedding_bag_weighted` reduce bags in
  C (see [embedding bags](#embedding-bags)).
- `meft_rows_last_*` report the previous call's block reads, decompressions,
  compressed and decoded bytes, and cache hits.
- Calls return 0 on success and write a readable message into `error` otherwise;
  discard the output after a failure.

IDs, sizes and pointers use fixed-width integers or `size_t`. A handle owns scratch
buffers and is not thread-safe: open one per worker and close it explicitly.

```bash
build/native_gather table.men 0 42 0
```

The C reader bounds-checks the header, descriptors and every queried ID, but does
not hash the whole file on open (the PyTorch adapter does). Only open artifacts
you trust, and keep external checksums for archival.

From Python, `NativeRows.rows(ids)` takes a list of Python integers and returns
bytes. `NativeRows.gather_array(ids, out=None)` takes a 1-D NumPy integer array and
returns an `(n, row_bytes)` uint8 array (or fills `out`); it passes both buffers
straight to `meft_rows_gather`. On uncompressed tables it halves lookup time
compared with `rows`
([profile](../../evidence/host_query_profile_20261112/README.md)).

## Native layout and construction

`MEFTNAT1` is a little-endian binary file: a versioned header, an optional
forwarding map (8 bytes per original row), 32-byte block descriptors, then block
payloads. Rows are opaque bytes; the caller interprets dtype and endianness.

| Version | Codecs | Forwarding map |
| --- | --- | --- |
| 1 | raw, zlib | yes |
| 2 | + XOR-with-previous-row then zlib | yes |
| 3 | same as 2 | no (row *i* is original ID *i*) |
| 4 / 5 | lz4 or zstd (explicit `codec=`) | yes / no |

The default export writes version 1–3; older readers reject 4–5. The current
reader reads all five.

```python
from ngram_store import pack_native

pack_native("table.raw", "table.men", row_bytes=512, block_rows=256,
            deduplicate=False, predictor="xor", max_block_bytes=8192)
```

- Each block is stored in whichever form is smallest: raw, zlib, or (with
  `predictor="xor"`) XOR residuals + zlib. Prediction restarts at the first row of
  each block. Raw fallback means a block never expands, but the map, descriptors
  and metadata can still make the whole file larger than the source.
- With no deduplication, reordering or representatives, the export streams the
  source block by block and writes version 3, saving exactly `8 × rows` bytes.
  Pass `implicit_identity=False` to write v1/v2 for older readers. On six trained
  Qwen tables, v3 with 32-row blocks saved 5.4–5.5% against raw FP32, with no query
  speedup ([report](../../evidence/identity_format_evidence_20261109/README.md)).
- `deduplicate=True` stores byte-identical rows once and needs an in-memory index
  of unique rows.
- `row_order=` takes a full permutation of original IDs for locality experiments.
  The forwarding map restores original IDs. Validation uses O(n) memory and the
  source is read out of order.
- The table keeps every row; nothing is pruned for a particular query trace.

`pack_native` writes the table to a sibling temporary file, fsyncs it, and
atomically renames it into place. On failure the old file stays. Open handles keep
reading the old inode. Writers are not coordinated, and the rename covers only the
table file, not the surrounding metadata.

Codec development history: [XOR blocks](../../evidence/native_xor_compression_20261020.md),
[streaming export](../../evidence/streaming_export_20261021.md).

## Blocks and cache

A query groups requested IDs by block, decodes each block once, copies the rows
out and restores request order. Rows per block are
`min(block_rows, max(1, max_block_bytes // row_bytes))`; a single row larger than
`max_block_bytes` gets its own block. Smaller blocks make sparse misses cheaper
and compression worse. On 1.85M trained rows, smaller blocks cut miss latency
16–27× against the earlier layout; raw mmap stayed faster
([report](../../evidence/sparse_blocks_evidence_20261030/README.md)).
`max_block_bytes` is accepted by the CLI, all exporters, `StructuralPlan.export`
and `ngram-store tiny-export`.

The cache is off by default. With a byte budget it keeps decoded blocks in an LRU;
`native_cache_policy="rows"` keeps only the queried rows instead, which suits
sparse repeated lookups. The budget covers cached payload; the hash index and LRU
links come on top, and it does not bound process RSS. Cache hits are served before
misses can evict them, and fully cached queries skip sorting
([measurements](../../evidence/native_query_evidence_20261030/README.md)).

Per-handle scratch holds up to one decoded block plus the current batch's request
locations. Already grouped IDs skip sorting.

## PyTorch

```python
from tools.engram_row_adapter import (
    export_demo_embedding, MappedMultiHeadEmbedding,
    export_embedding, MappedEmbedding,
)

# Engram-style multi-head memory: keep hash IDs and head offsets.
export_demo_embedding(original, head_sizes, "/tmp/memory-native",
                      backend="native", deduplicate=False)
memory = MappedMultiHeadEmbedding("/tmp/memory-native",
                                  native_library="build/libmeft_rows.so")
vectors = memory(head_local_ids_cpu)

# Ordinary nn.Embedding: keep global vocabulary IDs.
export_embedding(original_embedding, "/tmp/token-native", backend="native")
embedding = MappedEmbedding("/tmp/token-native",
                            native_library="build/libmeft_rows.so")
vectors = embedding(token_ids_cpu)
```

The adapters pass the ID tensor and the output tensor's storage straight to C, so
there is no per-row Python work. Output keeps shape and the exact bits of
float32, float16 or bfloat16 rows. IDs and outputs are CPU tensors; GPU use would
need explicit transfers.

- No `.weight`, gradients, tied output heads or full-model serialization.
  `nn.Embedding(max_norm=...)` is rejected because it mutates weights.
- Opening verifies a full-file checksum, so startup reads the whole artifact once.
- `query_chunk_rows` (default 65,536 row requests, counting each head) bounds the C
  sort scratch; `None` uses the single-call gather. It does not bound the output,
  staging tensors, cache or mmap.
- Modules support `copy.deepcopy` and pickling into `multiprocessing` spawn
  workers. Only configuration and small buffers are copied; each copy re-verifies
  the checksum and opens its own handle with a cold cache. The artifact and library
  must stay at the same absolute paths. A changed table fails verification. Close
  every copy. With fork, open modules inside each worker.

## Embedding bags

For inference with a sum/mean `nn.EmbeddingBag`:

```python
from tools.mapped_embedding_bag import export_embedding_bag, open_embedding_bag

export_embedding_bag(model.memory, "memory-bag", predictor="xor")
with open_embedding_bag("memory-bag") as memory:
    output = memory(ids)              # 2-D fixed bags, or memory(flat_ids, offsets)
```

Or wrap an existing single-head artifact with
`MappedEmbeddingBag(path, mode="mean", padding_idx=0)`. Output is `[bags, dim]`.

| Setting | Behaviour |
| --- | --- |
| `strategy="unique"` (default) | Gather each distinct ID once, reduce in PyTorch. Staging grows with query diversity |
| `strategy="chunked"` | Reduce in C, `chunk_rows=1024` at a time. Bounded staging; blocks may decode repeatedly |
| `unique_method="auto"` | Deduplicate IDs with a histogram when the table has ≤ 4 rows per queried ID and the index fits `unique_index_bytes` (8 MiB); otherwise sort. `"sort"` forces sorting |
| `per_sample_weights=` | Sum mode only; CPU tensor shaped like the IDs, in the row dtype. Each occurrence keeps its own weight; no normalization |

The C reducer accepts LE FP32/FP16/BF16 rows, nondecreasing offsets of length
`bags+1`, a padding ID (or `UINT64_MAX` for none), and accumulates in FP64 into
FP32 output, so rounding can differ slightly from PyTorch. Padding is excluded
from sums and counts; empty bags return zeros. No max mode, training or CUDA.
Dense in-memory lookup was faster in our tests
([report](../../evidence/embedding_bag_evidence_20261025/README.md)).

## Quantized rows

`tools.quantized_memory.pack_quantized_file` streams a headerless FP32 file,
encodes rows as FP32, FP16 or per-row INT8 (with an FP32 scale in each row), and
packs the encoded rows with the same native format:

```python
import numpy as np
from tools.quantized_memory import pack_quantized_file, QuantizedRows

pack_quantized_file("trained.f32", "int8-bundle", 256, "int8",
                    native="auto", chunk_rows=4096, max_block_bytes=8192)
ids = np.array([2, 0, 2], dtype="<u8")
with QuantizedRows("int8-bundle", "build/libmeft_rows.so") as rows:
    output = np.empty((len(ids), rows.dim), dtype="<f4")
    scratch = np.empty((len(ids), rows.width), dtype="u1")
    rows.gather_into(ids, output, encoded_scratch=scratch)
```

FP16 and INT8 change the weights; native storage then keeps the encoded bytes
exactly. `native="auto"` builds both a raw and a native file and keeps the smaller
one, which matters because INT8 rows often compress poorly. On 1.85M trained
rows, FP16 native saved 52.29% against raw FP32, while INT8 native was larger than
raw INT8 ([report](../../evidence/quantized_capacity_evidence_20261031/README.md)).
Chunking bounds codec and hash buffers; descriptors, optional indexes and
structural planning still grow with the table. The older in-memory
`pack_quantized(matrix, ...)` remains available.

## Model integration

| Model or runtime | Integration point | Done | Open |
| --- | --- | --- | --- |
| Engram-style memory | Multi-head gather after the model's own hashing | Official CPU module output matches in 6 random-weight cases, gates and ShortConv included ([report](../../evidence/engram_full_adapter_verification_20261018.md)) | Trained checkpoint, LM quality, GPU transfer |
| PyTorch decoder | `nn.Embedding` lookup | Byte-exact gather; small untied causal Transformer logits match | Tied heads, device placement |
| Qwen + TinyEngram branch | Trained n-gram branch | CPU checkpoint, full logits and greedy decode match ([report](../../evidence/tinyengram_qwen_evidence_20261030/README.md)) | Other architectures, GPU placement |
| FastText | `loadModel`, word/sentence vectors, classification via the [native facade](fasttext_native_bundle.md) | 90,380 trained rows byte-identical ([report](../../evidence/fasttext_native_evidence_20261031/README.md)) | — |
| llama.cpp-style C++ runtimes | Row gather behind a tensor op | C ABI compiles and passes tests | Loader and scheduler integration |
| vLLM / TensorRT-LLM | Host gather + prefetch to GPU | Not started | Custom ops, pinned memory, async queues |

Storing an existing table is a drop-in change. Adding n-gram memory to a model that
has none is an architecture and training change outside this library.

### N-gram order

Higher n-gram orders mean longer keys, more distinct n-grams, larger tables and
more lookups per token. In the [order sweep](../../evidence/ngram_order_native_20261018.md)
(maxima 2–12, unchanged official hash/gather code), shared-prefix key encoding
gained at higher orders, while random weight tables still grew after packing and
gather work increased. On trained SGNS tables (orders 4/8/12, three seeds),
structurally paired rows that receive identical updates have median cosine about
0.996, and their share grows with order; block storage saved about 5% losslessly
([report](../../evidence/trained_ngram_similarity_20261018.md)).

## Tests

Native tests compile pure C and check byte equality, repeated batch order,
raw/compressed blocks, output capacity, corrupt maps and descriptors, invalid IDs,
and truncated or malformed files. ASan/UBSan run on valid files and 64
deterministic mutations. PyTorch tests cover all dtypes and a small causal decoder.
The official Engram module check runs unchanged hashing, gates, projections and
ShortConv on a fixed token fixture across Python/native and three dtypes.
