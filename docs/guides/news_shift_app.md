# Open-vocabulary CPU news routing with precomputed/subword hybrids

The classifier can precompute every training-word feature and retain a
compressed n-gram fallback for new words. All four hybrids use the same FP32
known-word features; only their fallback differs. Full requests still perform
tokenization, known/OOV lookup, word/document means, ReLU and the classifier
head, without importing Torch.

The [independent public-news experiment](../../evidence/news_shift_evidence_20261105/README.md)
measures **37.84 MiB** of complete files and **38,264 documents/s** at batch 32
for the NGramStore INT8 hybrid on an Apple M5. It is 37.79% smaller than the
ordinary INT8 hybrid with 0.62% lower measured throughput. These are same-machine
three-seed medians; projection/quantization change original rows.

**The application-necessity hypothesis did not pass.** Zero-OOV precomputed
words remain smaller/faster and pass the fixed quality gates. Retaining subwords
has positive accuracy point estimates, but the paired intervals include zero,
including the 1,132 naturally OOV-heavy articles. Source-model transfer accuracy
is only 65.87% median. This is a backend/robustness comparison, not a deployed
service or universal advantage. Read the complete report before quoting results.

The supplemental audit also gives SQ8, PQ8x8 and clustering the same known-word
path. PQ/clustering hybrids are 28.22/24.41 MiB and pass every seed's task-quality
gates, while running faster than the NGramStore hybrid. Its closer source
predictions remain a separate fidelity tradeoff.

## Serve an exported hybrid

Install this checkout; public v0.4.0 predates these additions:

```bash
python -m pip install '.[incidence]'
ngram-store classify \
  --store experiment/hybrids/seed-3/precomputed-hybrid-ngram-int8 \
  --text 'Researchers unveiled a new computer chip.'
```

```python
from ngram_store import TextClassifier

with TextClassifier("experiment/hybrids/seed-3/precomputed-hybrid-ngram-int8") as model:
    result = model.predict(["The football team won a championship match."])
    print(result["predictions"], result["oov_counts"])
```

Backends: `precomputed-hybrid-source`, `precomputed-hybrid-int8`,
`precomputed-hybrid-faiss-sq4`, `precomputed-hybrid-ngram-int8`. The FAISS hybrid
requires faiss-cpu; the NGramStore hybrid needs NumPy and the native C library.
Additional controls: `precomputed-hybrid-faiss-sq8`,
`precomputed-hybrid-faiss-pq-8x8`, `precomputed-hybrid-faiss-cluster-8`.
Known words read identical source FP32 features. New words use available
**already-trained** character n-grams. No online training, evaluation-word
precomputation or dynamic word cache; missing n-grams add no learned information.

Hybrids remove whole-word rows and unused word-bag arrays from the fallback,
rebasing its n-gram suffix IDs. This is an application bundle contract; engine
original-ID ABI, legacy imports and previous classifier bundles stay compatible.
INT8 hybrids have no source-prediction certificate. Bounded FP32 retains its
separate flags/refusal behavior. Native decoded cache defaults to 4 MiB.

## Reproduce the independent comparison

Reuse frozen AG News checkpoints and **complete** exports for seeds 3/7/11:

```bash
python examples/reproduce_news_shift.py \
  --training ag-experiment/training \
  --existing-serving ag-experiment/serving \
  --ag-data ag-experiment/data \
  --output experiment
```

`--existing-serving` contains `seed-3`, `seed-7`, `seed-11` directories.
`--huffpost /path/to/data.json` reuses the pinned download. This command
explicitly downloads research data when that option is omitted. NumPy and
FAISS are needed for the complete comparison; Torch is only needed to reproduce
source training. The earlier single-seed [AG News entry](news_classifier_app.md)
must be assembled for all three seeds first. Exact source-training/export/run
commands are retained in the [evidence folder](../../evidence/news_shift_evidence_20261105/commands/).
The reproduction runs the 15 primary controls, then the three supplemental
FAISS hybrids; the latter retain their explicit post-primary audit role.

The [protocol](../../evidence/news_shift_protocol_20261105.md) amended an input-only temporal
split before predictions because World News was absent from calibration. The
final split uses per-category content-hash 20% calibration / 80% evaluation;
**it is not a chronological holdout**. All HuffPost articles are independent of
source training; none trained the classifier, registry or codebooks.

```bash
python -S docs/news_shift_evidence_20261105/verify.py
```

The standard-library verifier checks input/model hashes, the split, all task
metrics, frozen choices, budgets and saved paired document-bootstrap draws.
No article text, lexicons or weights are redistributed. The experiment measures
incidence projection, native packing and shared word precomputation; it does
not isolate DATrie or mark-compact lookup speed.
