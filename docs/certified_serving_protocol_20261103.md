# Ready-to-query certified serving protocol

Saved before CPU measurements of the new complete serving flow. Reuse the
existing 1,855,907 × 256 FP32 trained table, full 142,308-bag CSR and 512-bag
native trace from the 2026-11-02 study, retaining their hashes. No new data fetch,
large-model training, GPU inspection or process operations.

Run two fresh single-threaded CPU build workers: current legacy incidence fit,
FP32 group means, expanded canonical staging and marked raw native pack; and the
64 MiB managed-buffer external fit/means/direct native export. Both produce
only `table.men`, `domain.npz` and a manifest. The native format, group numbering,
block width (8 rows), raw codec, logical IDs and projected means must match.
Do not include native C verification or resident runtime materialization in
either construction worker. Record those separately in a third process.

Measure complete wall time, process peak RSS, staging logical bytes, final
table/certificate/manifest bytes, source code/input/output hashes and environment.
The earlier 33.06× fit/means RAM result remains its own narrower measurement;
do not transfer it to the complete pipeline. Budgets do not cap RSS or OS cache.
The source/file cache and CPU are shared/warm, with no exclusive/cold-cache,
energy or monetary claim. Preserve failed/slower runs.

Compare every native table byte, partition summary and mapping hash against
the fresh baseline and earlier raw native artifact. In the separate verifier,
check table integrity, load the legacy reader, compile the same original-ID
trace, compare every plan array and native sum/mean output, and require refusal
of an incomplete multi-member group. Also exercise the new PyTorch CPU module
on that actual trace: compare against the native reference bytes and measure
rounding differences from source FP32 `nn.EmbeddingBag`. No inference-speed or
downstream/LLM-quality claim follows from this operator integration.

The offline release example trains actual SGNS on repository-authored MIT text
using NumPy and independent initialization. It demonstrates installation and the
full build/query/refusal flow. Its tiny text is not a representative capacity,
quality, convergence or performance benchmark. Existing public-corpus weights
and corpus caches remain unredistributed. An independently runnable entry is
not an independent third-party reproduction or evidence of adoption.
