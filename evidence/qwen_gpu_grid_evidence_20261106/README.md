# Trained Qwen memory: faithful storage, faster hot queries, failed necessity gate

Eight A100s completed **nine order × seed training runs**, three fixed lower-rate
diagnostics, **63 same-source codec quality comparisons**, and **69 separate
full-inference CUDA workers**. The useful shipping change is a cache policy:
at the same 1 MiB native budget, caching queried rows instead of whole decoded
blocks speeds repeated 128-token prefill **1.325–1.388×** and uncached generation
**1.340–1.381×** across three seeds. Native FP32 outputs remain byte-exact.

**This does not establish a necessary LLM-memory backend or a win over FAISS.**
The large trained branches worsen held-out PPL compared with bypassing the
branch; lower-rate training restores a small branch benefit, but head constants
remain slightly better. FP32 native table saving stays below 0.26%, whole-model
files barely change, and raw quantization remains smaller than encoded native
quantization on these tables. These results preserve the failed qualification
instead of turning integration correctness into a model-quality claim.

## Training and identity controls

The reference uses local pretrained **Qwen3-0.6B**, pinned
[TinyEngram](https://github.com/AutoArk/TinyEngram/tree/317f23832aa8f8e382a0916ec469e81ba12b6dff),
one learned branch at layer 26 of 28, and actual TinyShakespeare text. This is
neither released DeepSeek Engram training weights nor a converged foundation
model. Backbone/tokenizer pretraining overlap with this public corpus is unknown.

The complete corpus is split by characters before tokenization: 80% training,
10% calibration, 10% test. All nine main models train on 1,024 evenly sampled,
nonoverlapping 128-token windows across training, batch four, four epochs,
**1,024 Adam steps / 520,192 target exposures per seed**. Final checkpoints are
fixed; test results select no checkpoint, seed, order or threshold. Each split
scores 64 windows / 8,128 targets. Nine models yield **146,304 checkpoint-target
evaluations over 16,256 unique text targets**; the same targets and repeated
control forwards are not extra independent data.

Only the branch trains with ordinary full-vocabulary next-token CE. Frozen
prefix reuse avoids repeating the first 26 decoder layers during training.
It uses CUDA FP32 arithmetic and CPU FP32 caches: about 512 MiB training plus
32 MiB per held-out split. Complete/cached logits and all branch gradients
match before training; frozen backbone SHA remains unchanged. Held-out scoring
uses the matched suffix schedule, with a separate complete-forward check on the
first window of each split. It does **not** rerun complete Qwen on every quality
window. These caches and dense-source quality oracles are not deployment-memory
measurements.

All seeds are 3/7/11; per-row dimension is 32. Changing order also changes table
capacity, query fan-out and branch projection width, so this is not a controlled
causal test of order alone. The source must beat **both** memory-off and the
unweighted mean of each original head by ≥1% relative PPL on calibration, with
positive lower paired-window excess-NLL confidence bounds. Test independently
checks the same rule. Bootstrap uses 10,000 paired-window resamples, seed 731;
text dependence and three training seeds limit population inference.

| Maximum order | Actual FP32 rows | Raw rows MiB | Mean CE of final 32 training batches, seed range | Test source PPL, seed range | Test memory-off PPL | Test head-mean PPL, seed range | Qualified seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 49,192 | 6.005 | 3.309–3.345 | 136.60–137.81 | 102.36 | 192.69–198.01 | 0/3 |
| 4 | 114,712 | 14.003 | 2.433–2.466 | 249.58–266.95 | 102.36 | 614.84–622.74 | 0/3 |
| 6 | 245,860 | 30.012 | 1.489–1.519 | 461.43–468.44 | 102.36 | 943.60–1009.19 | 0/3 |

Training fit improves as capacity grows, while held-out performance deteriorates.
This is consistent with overfitting/poor frozen-backbone adaptation; the sweep
does not isolate its cause. Native exports preserve **every source row and head
offset**, including unused rows, and reproduce all matched-schedule logits,
zero KL and zero changed predictions on calibration and test.

The exploratory lower-rate diagnostic was fixed after the main failure:
order four, all three seeds, one epoch / 256 steps, memory LR 0.0003 and branch
LR 0.0001. Its test uses the **next 64 previously unscored windows**; calibration
is reused and the original tested windows are excluded. It adds 48,768
checkpoint-target evaluations and 8,128 unique test targets. Memory-off test
PPL is 91.5206; source PPL is 90.7566–90.8069, with head-mean PPL
90.6681–90.7120. The branch helps about 0.79–0.84%, and constants improve it by
another 0.096–0.104%. All three still fail the fixed identity/branch gate.
This is a diagnostic, not independent confirmation of a hypothesis selected
from the first test.

## Codec competition: PPL preservation differs from output fidelity

FP16, symmetric per-row INT8, FAISS SQ8/SQ4, PQ-8×8, full-vector 256-centroid
clustering and OPQ-8×8 encode the **actual final trained tables**. FAISS fitting
uses at most 16,384 deterministic sampled rows queried by training windows.
Every allocated row is encoded, including unused initialization. Original
hash IDs, layouts, other weights and schedules remain fixed; there is no
post-compression retraining. Raw/native quantization reconstruct the exact same
quantized FP32 bytes, including each INT8 row's FP32 scale.

The comparison runs even though model necessity failed, as a fidelity and
codec-control diagnostic. Candidate vectors are fully reconstructed into GPU
memory for the quality oracle. Codec table sizes below include codebooks,
encoded arrays, row maps and codec metadata, **not whole-model deployment files**.

| Codec, primary order-four models | Median backend MiB | Test PPL change versus source, seed range | Changed next-token predictions, seed range |
| --- | ---: | ---: | ---: |
| Raw FP16 / native FP16 | 7.002 / 7.394 | −0.00048% to −0.00029% | 0–0.012% |
| Raw per-row INT8 / native INT8 | 3.939 / 4.755 | −0.0114% to +0.0038% | 0.172–0.283% |
| FAISS SQ8 | 3.502 | −0.0783% to −0.0210% | 0.541–0.714% |
| FAISS SQ4 | 1.751 | −2.575% to −1.220% | 7.542–7.899% |
| FAISS PQ-8×8 | 0.907 | −1.002% to +0.938% | 12.279–12.795% |
| FAISS OPQ-8×8 | 0.911 | −0.998% to +1.353% | 12.869–13.214% |
| Full-vector cluster-8 | 0.141 | +29.958% to +43.183% | 36.331–36.491% |

The fixed 0.1% / 1% PPL gates use the **upper calibration bootstrap bound**;
choose the smallest eligible backend on calibration only, then check test
without reselection. SQ4 is selected for all primary seeds at both budgets;
all confirm their PPL gates on test. On order three, one OPQ calibration choice
fails test confirmation and is retained. On order six, PQ is smaller and passes
PPL gates. These are strong competing controls, not NGramStore victories.

PPL improves for some lossy codecs because the source itself generalizes poorly.
This does not make them lossless: order-four SQ4 test mean KL is 0.0159–0.0171,
PQ 0.0461–0.0478 and OPQ 0.0463–0.0486. Full distributions, KL tails, accuracy
counts and prediction changes remain in the per-target reports. PPL gates
alone cannot certify faithful source behavior.

Actual native FP32 memory directories save only **0.115–0.255%** versus raw row
bytes. Complete FP32 bundles are about 2.41–2.43 GB; orders three/four expand
slightly and order six saves less than 0.001% of the complete checkpoint.
FP16 native expands about 5.6% versus raw FP16 and INT8 native about 20.7% versus
raw row INT8. Include these regressions when choosing a backend.

## Independent full CUDA serving and cache-policy improvement

Five fixed models—three primary seeds plus order-three/order-six seed three—run
12 independent workers each: resident FP32, staged dense FP32, native FP32,
four raw/native quantized controls and five FAISS codecs. Every worker measures
32/128-token prefill and 16-token cached/uncached greedy generation, batch one,
two warmups and seven repeats. Timed generation includes prompt work, host
hashing and per-token greedy ID readback; it is not steady-state decoder-only
throughput. Full output hashes retain provenance without publishing token IDs.

Resident/dense/native use the same host hash/history/prefetch schedule and
their same-schedule outputs match byte for byte. Codecs change rows, so their
output equality is measured rather than asserted. Quality selection occurs in
the separate calibration oracle, never in serving timing.

Nine additional workers compare native block/row caches against resident on
the three primary seeds, at fixed 1 MiB budget and two 4,096-row staging slots:

| Request | Native block-cache median ms | Native row-cache median ms | Paired speedup range across seeds |
| --- | ---: | ---: | ---: |
| 32-token prefill | 79.42 | 74.37 | 1.050–1.219× |
| 32-token prompt, 16 cached tokens | 1,095.64 | 1,063.49 | 1.017–1.042× |
| 32-token prompt, 16 uncached tokens | 1,283.41 | 1,208.31 | 1.058–1.067× |
| 128-token prefill | 106.28 | 77.56 | 1.325–1.388× |
| 128-token prompt, 16 cached tokens | 1,119.68 | 1,068.52 | 1.024–1.049× |
| 128-token prompt, 16 uncached tokens | 1,691.03 | 1,237.93 | 1.340–1.381× |

Each cell is the median of three per-seed medians, not pooled timing samples.
The row policy brings these warm paths close to controlled resident/dense
costs; it does not establish a universal speed win. Requests repeat one prompt,
so queried-row locality is favorable. Cold, varied or distributed workloads
are unmeasured. Torch allocation drops by roughly 14–15 MiB against resident,
under 0.7% of complete inference allocation; driver/context memory is separate.
CPU RSS and startup peaks are recorded, and a 1 MiB cache is not an RSS bound.

The production loader now chooses queried-row caching for CUDA when the caller
supplies a nonzero cache budget. CPU/MPS/zero-budget defaults retain blocks;
`native_cache_policy="blocks"` preserves the previous CUDA behavior. Both full
and incremental inference are verified. This reuses the existing native row
cache; the contribution here is the measured CUDA policy/default, not a newly
invented caching algorithm.

FP32 native workers construct the model without a dense source table or prefix
cache. **Codec benchmark workers still use the original FP32 bundle for
startup/layout validation**, then close its table before queried-code serving.
Their on-disk requirements include that original bundle plus codec files;
they are evaluation adapters, not standalone deployable compressed Qwen bundles.
Do not claim whole-model codec file savings or source-free codec startup from
their smaller standalone table sizes.

Each timing process gates on an empty selected device and checks competing
compute clients around timing. These sampled checks do not reserve hardware;
CPU/I/O and the other GPUs are shared. The scripts handle UUID selection and
container/host PID differences without ignoring unknown clients. No training
or evaluation job was killed. Verified matmul-only occupancy jobs were released
under user authorization; task-specific release helpers end with the experiment.

## What the actual prefix/suffix analysis says

The nine final trained tables are also examined using actual observed compressed
token n-gram keys. Within each head, rows with multiple observed training keys
are excluded; eligible rows need at least two observations. One deterministic
pair per shared-prefix/suffix group is compared with unrelated **actual rows**
in the same head. No random vectors stand in for learned compression evidence.

No sampled pair passes the joint cosine/norm-ratio 0.99 guard. Collision
exclusions leave only small, selected subsets, and some heads have no eligible
relation pair. This does not refute the separate additive-SGNS equal-incidence
evidence: position-wise gated TinyEngram concatenation and hash buckets do not
certify shared updates. String relations and “in the same training run” alone
are insufficient to assert 99% vector similarity. Training/initialization and
the model's actual update incidence remain necessary context.

## Reproduce, inspect, and audit

Use the pinned protocols, local model/corpus and source from the recorded
commits. Exact commands, device UUIDs and environment are in every raw report;
the archived [training launcher](training/run.sh), [codec launcher](codecs/run.sh),
[serving launcher](controls/serve.sh) and [cache launcher](controls/cache.sh)
show shard scheduling. Do not reuse their internal paths or launch occupancy
cleanup when adapting the protocol to another host.

```bash
python -m tools.train_tinyengram_identity --device cuda --seed 3 \
  --base-model /path/to/local/Qwen3-0.6B --corpus /path/to/corpus.txt \
  --upstream-dir /path/to/pinned/TinyEngram --native-library /path/to/libmeft_rows.so \
  --protocol docs/qwen_identity_gpu_protocol_20261106.json --output-dir /new/training

python -S docs/qwen_gpu_grid_evidence_20261106/summarize.py
python -S docs/qwen_gpu_grid_evidence_20261106/verify.py
```

[Summary](summary.json), [protocols](protocols/), [raw training](training/),
[codec reports](codecs/), [diagnostics](diagnostic/), [serving](serving/),
[cache controls](cache-serving/), [relations](relations/),
[runtime snapshots](runtime/) and [file manifest](manifest.json) permit inspection
without redistributing corpus text, vocabulary, token IDs or model weights.
The dependency-free audit recomputes deterministic metrics, pairing, fixed
decisions and timing medians; bootstrap generation requires NumPy and its saved
seed. [Actual file inventory](inventory.json) counts all memory/complete bundle
costs. [Environment](environment.json) records compiler/library hashes and
hardware/software.

Two original evaluation failures—CUDA ID bridging and JSON tuple/list replay—
remain in [superseded](superseded/), along with their successful repair tests.
The initial new codec staging test caught a CPU output-shape error; its failed
log and subsequent [26-test CPU/CUDA pass](controls/fixed-tests.log) are retained.
[Final codec tests](controls/final-codec-tests.log) check actual FAISS original-ID
forwarding, and [CUDA default tests](controls/default-cuda-tests.log) pass full
history, generation and codec staging. Local optional-dependency skips and
Linux/macOS CI are kept separate.

Environment: 8 × A100-SXM4-80GB, AMD EPYC 7742, Linux x86-64, Python 3.11.15,
Torch 2.11.0+cu130, Transformers 4.57.3, NumPy 2.2.6, FAISS 1.15.0. Training
and fitting use eight CPU threads; serving uses one Torch/interop thread.
FP32, deterministic algorithms, no autocast/TF32, local cached data/models,
no new download or provider fee. GPU model/driver details and original tokenizer
warnings remain in the raw artifacts. All experiment and cleanup workers exit.

The next model-quality experiment needs a genuinely useful independently
trained memory whose row identity survives the head-constant control, adequate
training coverage and joint model/memory learning or an independently supplied
checkpoint. Enlarging this frozen late-layer branch alone does not meet that
requirement. Storage fidelity and the measured query-policy improvement stand
on their own, with the failed application-necessity result alongside them.
