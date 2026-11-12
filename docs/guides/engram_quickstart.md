# Engram: NGramStore as LLM n-gram memory

Engram-style models look up token n-gram rows with several hash heads and mix
them into the hidden state through gates and a short convolution. NGramStore can
take over the table behind `MultiHeadEmbedding`: the model keeps its hash IDs,
head offsets, projections and gates, and the stored rows come back byte for byte.

## What has been checked

| Goal | Start here | Status |
| --- | --- | --- |
| Run the official Engram module with native storage | [CPU forward check](../../evidence/engram_full_adapter_verification_20261018.md), [XOR codec follow-up](../../evidence/native_xor_compression_20261020.md) | Full module output matches in nine cases (Python / native / native-XOR × three dtypes), tiny random weights |
| Open a frozen table without first allocating the dense one | Factory example below | Same nine cases; default allocation is blocked and never reached |
| Convert a trained TinyEngram/Qwen checkpoint | [checkpoint evidence](../../evidence/tinyengram_qwen_evidence_20261030/README.md) | Trained CPU/CUDA outputs match; native lookup is slower than dense offload on these traces |
| Measure whether the memory is needed at all | [evidence §1](../../evidence/README.md#1-does-exact-n-gram-identity-beat-hash-buckets) | Mixed: see the necessity-gate results |
| Another model or runtime | [C ABI reference](native_ngram_memory.md#model-integration) | Same backend; model-specific addressing stays in the model |

None of this uses a released, trained DeepSeek Engram checkpoint; there was none
to test.

## Factory integration example

The upstream module allocates its dense table in the constructor. A small change
proposed in [deepseek-ai/Engram#27](https://github.com/deepseek-ai/Engram/pull/27)
(open, not merged as of 2026-11-12) adds an optional `embedding_factory`, so a
frozen store can be passed in instead. The example below uses a pinned copy of
that proposal.

```bash
python -m pip install torch numpy sympy transformers tokenizers
make native        # macOS: make native NATIVE_SHARED=-dynamiclib NATIVE_EXT=dylib
curl -fL https://raw.githubusercontent.com/initial-d/Engram/8b4155c2f8fc886ef27973d1164b798126f90c0d/engram_demo_v1.py \
  -o /tmp/engram_factory_v1.py
python tools/demo_engram_factory.py \
  --proposal-source /tmp/engram_factory_v1.py \
  --native-library build/libmeft_rows.so \
  --output-dir artifacts/engram-factory
```

The script checks the proposal's SHA-256 before importing it, builds tiny
random-weight fixtures, exports each table, and constructs a second Engram module
on top of the native store with dense allocation patched to fail. It then runs
the original hashing, projections, gates and ShortConv and compares the full
output, for float32/float16/bfloat16 × Python/native/native-XOR. Tokenization
uses a small identity fixture; nothing is downloaded besides the source file.
Recorded runs: [six cases on 2026-10-19](../../evidence/engram_factory_verification_20261019.json),
then [all nine including native-XOR on 2026-10-20](../../evidence/native_xor_compression_20261020.md).

Without the proposal, the current upstream module still works through
post-construction replacement ([example](../../evidence/engram_full_adapter_verification_20261018.md));
it just allocates the dense table first.

## Use your own table

Export each layer's table with `export_demo_embedding`, then hand the loader to
the module:

```python
from functools import partial
import torch
from ngram_store import open_demo_embedding

factory = partial(open_demo_embedding, "/path/to/exported/layer-1",
                  native_library="build/libmeft_rows.so")
engram = Engram(layer_id=1, embedding_factory=factory)  # from the proposal source
try:
    # load this layer's non-embedding weights from the checkpoint here
    engram.eval()
    with torch.inference_mode():
        output = engram(hidden_states, input_ids)
finally:
    engram.multi_head_embedding.close()
```

- The loader checks head layout and dimension and refuses a mismatch, including a
  different split of the same total row count.
- It restores only the table. Projections, gates and tokenizer settings come from
  your checkpoint.
- `.to(dtype=...)` does not re-encode stored rows; export in the dtype you serve.
- Inference only, on CPU, one handle per layer and worker. For GPU, see the
  [device staging adapter](device_memory_prefetch.md) and the
  [Qwen CUDA timings](../../evidence/qwen_cuda_performance_20261030/README.md).

## Compression on n-gram tables

Measured on trained character n-gram tables, not on Engram weights:

| Mode | Saving | Vectors |
| --- | --- | --- |
| Native block storage | ~5% | Unchanged |
| [Adaptive XOR layout](../../evidence/native_xor_compression_20261020.md) | 6.9–9.1% (layout needs training signatures) | Unchanged |
| Joint sharing, cosine and norm ratio ≥ .99, max order 12 | 2.14–2.16× | Mean n-gram relative L2 error ~0.48% |
| Joint sharing, cosine ≥ .98, norm ratio ≥ .99 | ~2.35× | Mean ~0.52%, worst ~3.79% |

([trained study](../../evidence/trained_ngram_similarity_20261018.md),
[distortion curves](../../evidence/approximate_ngram_compression_20261018.md))

Engram tables are addressed by hash. A bucket mixes unrelated n-grams, so
prefix/suffix sharing does not apply to them directly; use plain `pack`. Vector
error is not the same as loss in accuracy or perplexity, which has to be measured
on the full model.

## Reporting a result

Include the table's source and hash, layer and head layout, dtype, hash and
tokenizer settings, and the original query IDs. Compare total artifact bytes,
build memory, RSS and cold/warm gather latency against a resident tensor and raw
mmap. For approximate modes, add per-row error distributions and held-out model
quality. Use the
[benchmark template](https://github.com/initial-d/ngram-store/issues/new?template=benchmark-result.yml)
or [ask a question](https://github.com/initial-d/ngram-store/issues/new?template=evaluation-question.yml).
Citation: see the [citation guide](citation_guide.md).
