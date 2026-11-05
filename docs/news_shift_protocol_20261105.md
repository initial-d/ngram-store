# Open-vocabulary news routing: frozen follow-up protocol

This follow-up tests whether retaining a subword backend matters on naturally
newer news, and whether incidence compaction can make that backend useful under
a complete deployment-file budget. Freeze this protocol before any new task
predictions. A positive result is not required. No GPU use or inspection.

## Independent input and backend-calibration split

Use the HuffPost News Category Dataset mirror `heegyu/news-category-dataset`,
revision `304a05a55bc6abc0446d8fae0d0771716b6a271a`, `data.json`. The mirror declares
CC BY 4.0 and points to Rishabh Misra's original Kaggle dataset. Record the
download URL, SHA256, dates, original attribution and mirror license notice.
Do not publish article text, links, lexicons, arrays or weights in this repository.

Keep only the literal source categories `WORLD NEWS`, `SPORTS`, `BUSINESS`,
`TECH`, mapped respectively to the existing World/Sports/Business/Sci-Tech
outputs. These taxonomies are related but not identical: do not relabel politics,
science, entertainment or other categories to obtain a bigger favorable set.
Text is the original headline plus short description. Existing lowercase ASCII
tokenization, apostrophes and 128-token maximum remain unchanged. Drop inputs
with zero supported tokens, recording the count. Sort by date and original line
ID; retain the earliest occurrence of each tokenized input and record duplicates
and conflicting-category duplicate counts. Exclude exact tokenized matches with
the AG News update/validation inputs, if any, and record them.

An input-only audit of the original pre-2017/post-2017 design found **zero World
News calibration articles** (11,400 calibration / 5,015 evaluation inputs).
Before any external predictions, amend that design to a stratified 20% backend
calibration / 80% evaluation split. Preserve the rejected input audit; do not
claim chronological held-out evaluation. Within each literal category, sort
deduplicated inputs by SHA256 of their tokenized text (original line ID breaks
ties) and assign the first floor(n/5) to calibration, all others to evaluation.
This uses category labels only for stratification, never model output. The
AG News source training is still independent of every HuffPost article.
Publish numeric request
IDs, labels, dates, content hashes and preprocessing counts, never article text.
No artificially corrupted text, held-out training types or injected new names.

## Frozen models and mandatory strong controls

Prefer all three existing nonlinear AG News checkpoints (seeds 3/7/11), the
selected epoch, 64-dimensional word/document means, ReLU and four-class head.
If the remote artifacts are unavailable, reproduce the original frozen AG News
training protocol locally (same full 108k/12k split, seeds, eight epochs,
validation-CE checkpoint selection and settings), before any external-news
predictions. Declare these as new local source checkpoints with their own
platform and hashes; do not mix them with the original remote results. This
fallback is frozen before external predictions and never trains on HuffPost.
No target-domain retraining, external-data codebook fit, changed registry, calibration or
test-driven model settings. Retain source/checkpoint hashes. All ordinary
backends from the original experiment remain included: source-resident/mmap,
FP16, INT8, FAISS SQ4/SQ8/PQ8x8/clustering, bounded FP32, NGramStore INT8,
and zero-OOV precomputed training words.

Add four **precomputed + subword fallback** controls: source FP32, ordinary
rowwise INT8, FAISS SQ4, and NGramStore incidence-projected INT8. Every hybrid
stores the *same* precomputed FP32 source features for every training word.
Unknown words gather available trained character n-grams; no vocabulary updates
or evaluation-word precomputation. Remove independently trained whole-word rows
and word-bag ID arrays from the fallback bundles: known words already have final
features and unknown words never use whole-word rows. Rebase all fallback gram
IDs consistently. Count full word vectors, gram keys, fallback codes, native maps,
scales/codebooks, classifier head and metadata. Reuse the ordinary SQ4 codebook
and slice its existing n-gram codes; do not fit it on a different sample. The
NGramStore hybrid reuses the same training-incidence groups and projected means.

All hybrids have the same fast known-word path. This prevents attributing
precomputation's speedup to NGramStore. INT8/projection modes change original
rows and do not have the bounded-FP32 prediction certificate. Keep certified
FP32 as a distinct, potentially slow control. Preserve legacy bundles/interfaces.

## Decisions and complete-request evaluation

Every nonempty request receives a prediction. Measure all calibration inputs,
freeze choices, then evaluate every evaluation input. Quality gates versus the
same seed's source-mmap: accuracy loss <=0.5 percentage points, macro-F1 loss
<=0.5 percentage points and CE increase <=2%. Fixed complete-file budgets:
16/32/48/64/96/128/256 MiB. Fixed observed-RSS budgets: 64/128/192/256/384 MiB.
Choose the fastest batch-32 calibration backend passing all quality/file or
quality/RSS gates. Audit the selected backend's evaluation quality and budget;
retain failures. These are eligibility checks, not OS/cgroup memory limits.

Isolated fresh NumPy/C worker for each seed/backend/phase; no Torch imports.
Full text -> tokenization -> known/OOV lookup -> word/document mean -> ReLU ->
head/output time. Native decoded cache 4 MiB. No hidden word cache. One CPU
thread, no exclusive/core-pinned, cold filesystem-cache, production or GPU claim.
Set OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS/VECLIB_MAXIMUM_THREADS
to 1, including Apple's Accelerate runtime. The first incomplete local run
omitted VECLIB_MAXIMUM_THREADS; retain it as superseded, then repeat all backend
calibration, freeze and evaluation measurements with the same models, inputs,
codecs and quality/budget thresholds. No result-dependent method tuning.
Fixed seed-42 sample of up to 512 calibration documents, batches 1/32, one warmup
and five repeats. Also record one full evaluation traversal's docs/s separately.
Linux RSS uses current-exec VmHWM, not the parent's inherited ru_maxrss.

Report all-input accuracy/macro-F1/CE, confusion matrix, paired source agreement,
source-preservation flags, OOV token rate, missing-ngram rate, truncation, and the
predeclared >=10% OOV slice. Publish paired accuracy-change confidence intervals
by resampling document IDs (2,000 bootstrap samples, seed 42), averaging the
three seeds per document before resampling; do not pretend the three predictions
for a document are independent data. Separate calibration choices and evaluation
audit. No strong OOV-specific quality claim for a slice smaller than 200 inputs.

An application-necessity claim requires a frozen budget where the NGramStore
hybrid passes evaluation quality and file limits while *all* other complete
subword-preserving hybrids fail at least one, and the zero-OOV control fails
quality. Even then state the dataset/model/budget, explain the mechanism, and
report any throughput/RSS disadvantage. If these conditions do not occur,
publish the negative boundary instead of claiming universal superiority.
