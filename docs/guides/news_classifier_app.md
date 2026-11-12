# CPU news classification with a learned n-gram backend

This application uses the full official 120,000-document AG News training split
(108,000 for updates, 12,000 for validation), exports the learned
character-n-gram table and accepts actual text.
The four categories are World, Sports, Business and Sci/Tech. A 64-dimensional
deep-averaging model applies ReLU before its classifier head. Training uses
PyTorch on CPU; frozen inference needs NumPy and the native C library only.
This is a public-data application benchmark, not a state-of-the-art classifier.

For new-word serving, the [precomputed/subword hybrid follow-up](news_shift_app.md)
adds four fair fallback controls and an independent news corpus. Its storage
benefit and the failed application-necessity hypothesis are reported separately.

## Run the application

From a source checkout, with a C compiler and zlib development headers:

```bash
python -m pip install '.[incidence]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python examples/reproduce_news_classifier.py --output /tmp/ngram-news
```

This explicit command downloads hash-pinned research data, trains eight epochs,
selects the checkpoint by validation loss, exports five standalone bundles,
freezes validation-only budget choices and evaluates all 7,600 official test
documents. Ordinary INT8 and precomputed-word controls run by default alongside
the source and two NGramStore modes. It also predicts two actual texts. No GPU is used. The output directory
must be new; data and weights remain there under the source dataset's notice.
Full training and export need substantially more memory than frozen inference.

```bash
ngram-store classify --store /tmp/ngram-news/serving/ngram-int8 \
  --text 'The team won the final championship match.' \
         'Scientists announced a new computer chip.'
```

Or load the same bundle in an application:

```python
from ngram_store import TextClassifier

with TextClassifier('/tmp/ngram-news/serving/ngram-int8') as classifier:
    result = classifier.predict(['Scientists announced a new computer chip.'])
    label = classifier.classes[result['predictions'][0]]
    probability = result['probabilities'][0]
```

Classification never downloads anything. Every text with at least one supported
ASCII word receives a prediction; at most 128 tokens are retained. Unknown words
use available training n-grams. The result exposes token/OOV/truncation counts.
Probabilities are model outputs, without a calibration guarantee.

For the FP32 aggregate backend, `--store .../ngram-bounded --require-preserved`
refuses a batch if any prediction lacks a source-model agreement certificate.
The certificate propagates source-relative residual bounds through the word and
document means, ReLU and final head. It establishes prediction preservation,
not correctness against the news labels or preservation of individual row bytes.
INT8 has no such certificate. Both compact modes change original rows; ordinary
lossless table packing has a separate contract.

## Evaluate the alternatives

Add `--full-controls` and install `faiss-cpu>=1.8,<2` to include resident/mmap
FP32, FP16, rowwise INT8, FAISS SQ4/SQ8/PQ8x8/full-vector clustering, and
precomputed training-word vectors. Precomputed words use zero for OOV tokens;
this deliberately strong control can be smaller and faster. For a fully linear
model, folding the four-class head into the source rows is another strong control;
the ReLU graph prevents this algebraic folding.

The [complete three-seed evidence](../../evidence/news_classifier_evidence_20261105/README.md)
keeps both graphs and all controls. Model-file ceilings and observed process-RSS
ceilings are distinct, neither enforced by an OS memory limit. Backend selection
uses validation quality and speed; official test results only audit those choices.
See the frozen [initial](../../evidence/news_classifier_protocol_20261105.md) and
[nonlinear](../../evidence/news_relu_protocol_20261105.md) protocols for the full run settings.

The `classify` command is included from v0.5.0.
