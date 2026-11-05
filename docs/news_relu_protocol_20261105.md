# Nonlinear news task and mandatory linear-folding control

Freeze this follow-up before evaluating its test predictions. The initial
[protocol](news_classifier_protocol_20261105.md) chose a linear mean/head model.
Algebra gives a stronger application control: project every source row through
the four-class head offline, retaining all original/OOV n-grams with four
stored values instead of 64. FP32 materialization has a separate rounding audit.
This parameter-free control must not be omitted or confused with generic
64-dimensional row compression. Preserve the original linear runs and add its
fused-row control; do not use their test metrics to choose a new architecture.

The primary next application is a simple nonlinear deep-averaging classifier:
same word/subword means and document mean, followed by elementwise ReLU and the
four-class linear head. ReLU prevents pushing the head through document
aggregation. This is a standard class of embedding-based text models, not an LLM
or a claim that all classifiers require the vector backend. The source-model
accuracy remains its own benchmark, not a state-of-the-art claim.

All dataset/preprocessing/split/seeds/dimension/batches/LRs/eight epochs and
validation-only epoch selection remain fixed from the initial protocol.
Train separately with `--activation relu`, using the same complete AG News CSV
hashes and training-only registry. No test-derived graph, hyperparameter or
threshold selection. Three seeds, 3/7/11. The model/source/bundle directories are
separate from the preserved initial linear experiment.

All 11 original full-source/FP16/INT8/FAISS/native/precomputed-word controls are
retained in the nonlinear task. Linear head folding is explicitly inapplicable
to this graph. Precomputed words still give up unknown-word subwords and remain
a strong control. Identical validation quality gates, artifact and observed-RSS
ceilings, 512 validation requests, batches 1/32, one warmup/five repetitions,
all 7,600 test examples and the fixed >=10% OOV slice remain in force.

Prediction-preservation bounds apply after ReLU because it is 1-Lipschitz in L2
and `max(x,0)` is exactly evaluated on binary64 inputs. The existing document
deviation bound then encloses the activated-vector deviation. The final head
proof includes FP64 dot/margin rounding and FP32 source weights. The certificate
is source-real-model agreement, not ground-truth correctness or original-row
byte preservation. Preserve every uncertain prediction and every failed control.

Primary findings must state which task is nonlinear, show the linear-folding
counterexample, and report any precomputed-word or FAISS control that wins under
the same task/budget. No engineered universal-advantage or publicity claim.
