# Real news-routing application, frozen protocol (2026-11-05)

Build a CPU news classifier and replace its learned character-n-gram table,
then measure complete requests, task accuracy and deployment costs. No GPU
inspection/use, new language-model weights, or edits outside a task directory
under `/llm-align/duyimin/`. AG News CSVs are explicitly downloaded for this
application from revision `555590db4219b1243abb1918effd6a7425a2d75f` of
`mhjabreel/CharCnn_Keras`; preserve URL, SHA256 and its research-use notice.
Do not redistribute news text, training arrays or model weights in this repo.

## Data, model and training

- Full official 120,000 training documents; stratified 108,000/12,000 train/val,
  seed 42. Official 7,600 test documents untouched until model/backend selection.
  Labels: World, Sports, Business, Sci/Tech. Join title and description.
- Lowercase ASCII words with apostrophes, at most 128 tokens per document.
  Training-only word vocabulary, minimum count 2, no hash buckets/vocabulary cap.
  Exact bounded character n-grams, orders 2 through 8, from all training token
  types (including singletons). Include an independent whole-word row for words
  with count >=2; register singleton training words with their character bags
  too, so grouping covers every word type actually used for training updates.
  Unknown words use available training n-grams; no new types
  added at inference. Record OOV/missing-ngram/truncation counts.
- 64-dimensional embedding-bag word features (mean of word+subword rows), mean
  word features, a four-class linear head. Sparse SGD embeddings, ordinary
  classifier-head SGD; word-mean forward and unscaled word-bag backward match
  the explicit legacy FastText-style convention. This is a supervised Python
  reference, not a reproduction of upstream FastText's supervised implementation.
- Seeds 3/7/11, eight epochs, batch 128, embedding/head LR 1.0/1.0 with linear
  epoch decay; initial embeddings uniform(-1/64, 1/64), head normal(0, .1).
  Choose each seed's epoch by lowest validation cross-entropy, never test.
  No changes based on test. Report finite training, losses and source hashes.

## Frozen backends and fairness

The trained source/head and complete train-only key registry stay the same.
Build incidence groups from training word bags, protecting whole-word IDs.
Export NGramStore FP32 means plus the actual-source residual sidecar. For its
predictions, assess every word without discarding above-budget candidates; report
source-relative bounds and optional linear-head prediction-preservation flags.
A flag guarantees agreement with the declared source model in real arithmetic,
not agreement with ground-truth labels. Default classification returns every
nonempty request, with the flag separate; explicit require-certified mode refuses
uncertain requests and its coverage is reported.

Predeclared controls:

1. Resident source FP32 and raw mmap source FP32.
2. Direct rowwise FP16/INT8.
3. Direct FAISS SQ4, SQ8, PQ8x8 and 256-centroid full-vector clustering.
4. NGramStore bounded FP32, and native packing of INT8-encoded incidence means
   (an additional approximate route without the FP32 residual certificate).
5. Precomputed training-word vectors, using zero for unknown words. This strong
   smaller/faster control explicitly gives up subword OOV features.

Codebooks use a seed-42 sample of at most 32,768 actual checkpoint rows and 20
iterations, one FAISS thread; no validation/test rows train the codebooks.
All backends count their full table, keys, head, maps, scales/codebooks, sidecar
and metadata. Share no original-row fallback in compact variants. Library/runtime
costs and measured process RSS are separate from model files. Original-source
full-fidelity controls can remain faster or smaller in RSS; retain those results.

Fixed complete-model artifact budgets: 16/32/64/128/256 MiB. This is a deployment
file budget, **not an OS-enforced RAM cap**. Also report fresh-worker peak RSS and
startup costs; do not relabel mmap virtual/file bytes as resident RAM. Select the
fastest validation backend at each budget among those with accuracy loss <=0.5
percentage points and cross-entropy increase <=2% versus its source. Test only
audits these fixed choices, alongside all method comparisons. No winner is required.

## Evaluation and serving

All 7,600 official test documents: accuracy, macro-F1, cross-entropy, agreement
with source, confusion matrix and prediction-preservation coverage. An OOV-heavy
slice is fixed by >=10% unknown tokens using the train registry, without labels;
it is a real test subset, not generated corruptions. Include empty-slice results.
Retain per-example labels/predictions/losses/flags as numeric evidence only.

Evaluate quality in isolated NumPy/C workers without importing Torch; the resident
FP32 source is an explicit separate control. Measure full text -> tokenization ->
lookup -> word/document aggregation -> head/flags -> output, with no precomputed
test vectors, ID-only replay, source matrix retained in compact workers or hidden
word-vector cache. Native decoded row cache is fixed at 4 MiB and counted apart.
Latency: validation first 512 documents in fixed seed-42 order, batches 1 and 32,
one warmup and five repetitions; preserve raw per-batch timings, p50/p95,
documents/s and startup/peak RSS. Quality batching is 64 documents. Shared CPU,
one query thread, no exclusive/cold-cache, GPU, energy or monetary claims.

Provide a source-checkout `classify` command accepting actual text and a frozen
bundle, plus a reproducible download/train/export/evaluate entry point. Keep
historical imports, strict bag queries and file formats. Check installed-wheel
NumPy-only inference and exact-fraction classification certificate fixtures.
Publish metrics/source/environment/commands/hashes and failed runs; no data or
weights. Positive task evidence remains dataset/model-specific, not production
adoption, universal FAISS dominance or a publicity guarantee.
