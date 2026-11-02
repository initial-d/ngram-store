# Frozen-codec query-schedule retiming protocol

Saved before the 2026-11-02 run. Improve the scalar query certification in the
FAISS codec path using the existing batched integer certificate, and avoid
building a full fallback rank map when no query needs fallback. No codebook
refitting, new vectors, corpus fetch, quality reselection or GPU work.

Freeze the 0.1% loss-budget validation selections from the
[2026-11-01 comparison](faiss_cotraining_evidence_20261101/README.md), whose
uncompressed merged report hash is recorded in its analysis. All nine original
128-dimensional SGNS checkpoints participate: orders 4/8/12 and seeds 3/7/11;
direct, certified-group and full-domain hybrid, 27 selected variants total.
These all selected SQ6. Original model bytes and query domains remain fixed.

The reference is the complete benchmark module from source
`f9994ca2dbe15f54826123ee2b0e3f684da6e5ff`. Its `schedule` matches the scalar
function in the selected SQ6 rate phase (`64ebc742`); record module/function
hashes. The earlier base phase lacked the later encoded-byte-map option, which
is unused by these selected SQ6 candidates. This source clarification does
not change the saved run settings.
Require the reducer source to match the new module exactly. Load the existing
codec/codebooks and common row artifacts, verify trace hashes, and compare
every saved schedule array with old/new compilation. Also check every one of
the 1,200 registered known-word outputs byte-for-byte, and all measured known
and OOV trace outputs. Equality proves this path change adds no new distortion;
it does not make the codec lossless to the original FP32 model.

For each selected variant use a fresh CPU process, one FAISS/BLAS thread,
two warmups per function, 15 repetitions and a fixed seed-42 interleaved order.
Measure five modes separately: old compilation, new compilation, precompiled
replay, old compilation plus query, new compilation plus query. Retain samples,
p50/p95, accepted/nonempty coverage and original codec file hashes. Failed
workers remain in the report and make the benchmark command fail.

The shared CPU has warm pages; paired interleaving mitigates temporal drift but
does not establish exclusive service latency. The selected direct codecs are
unchanged controls. Peak RSS includes loading/output checks and runtime, so it
is not isolated query RSS. OOV rejections never count as completed work.
Precomputed known-word inference remains a stronger, narrower serving control.
Do not reuse old scalar timing numbers as new direct measurements or infer a
cross-domain speedup. Report any remaining slowdown against direct SQ6.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tools.benchmark_faiss_schedule \
  --report "$FROZEN_MERGED_REPORT" \
  --roots "$BASE_ROOT" "$RATE_ROOT" "$CLUSTER_ROOT" \
  --reference-module "$OLD_MODULE" --output-dir "$NEW_OUTPUT" \
  --loss-budget 0.001 --repeat 15 --warmup 2
```
