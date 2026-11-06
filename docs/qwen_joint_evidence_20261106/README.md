# Joint Qwen adaptation: independent cheaper controls still win

Three A100s completed **nine fixed final checkpoints**: full n-gram memory,
independently trained head constants, and independently trained memory-off
Qwen, for seeds 3/7/11. Unlike the earlier frozen-backbone experiment, all three
controls train the **same last two decoder layers**. Full memory loses to both
independent controls on every seed, on calibration and on previously unscored
test windows. Native FP32 reproduces the source bytes on all six memory models.

This strengthens the experimental controls, **not a claim that this late-layer
LLM memory needs a large storage backend**. Repeatedly enlarging or quantizing
this reference cannot establish that need. It also does not refute useful
memory architectures, the separately tested SGNS incidence mechanism, or the
existing native storage/query results on other applications.

## Fixed experiment and independently trained baselines

The [protocol](protocol.json), committed in runtime
`1b5033a57bb776a225bcd0046f192d4a94288454`, follows the earlier frozen-backbone
failure. It is an exploratory follow-up, with no test-based checkpoint, seed,
corpus, order or hyperparameter selection. Each model uses local pretrained
Qwen3-0.6B, pinned TinyEngram
`317f23832aa8f8e382a0916ec469e81ba12b6dff`, and the same cached public
[TinyShakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
These are not released DeepSeek Engram weights or foundation-scale training.
Qwen pretraining overlap with this corpus is unknown.

The complete corpus is split 80/10/10 by characters before tokenization.
All controls train on 1,024 evenly sampled nonoverlapping 128-token training
windows, batch four, four epochs, **1,024 Adam steps / 520,192 target exposures
per model**. All three controls of a seed use identical window order and
initial decoder weights. The final fixed epoch is evaluated. Decoder LR is
1e-5, memory LR 3e-4, other branch LR 1e-4; gradient clipping is 1.0.

Memory is at layer 26 of 28, maximum order four, vocab-size requests
8,191/16,381/32,749, two heads per order and 32 FP32 values per physical row.
The actual complete table has **114,712 rows / 14.003 MiB raw payload**.
The full model trains 35,534,080 parameters. The constant control trains
31,863,488 parameters, including **six independently trained 32-value vectors**
initialized from the corresponding original head means. It keeps the original
head addressing layout but ignores within-head row identity. The off control
trains 31,461,888 decoder parameters and bypasses the branch completely.
This is the same sampling and optimizer budget, **not equal parameter counts**.

Calibration uses 64 windows starting at index 64; test uses 64 windows starting
at index 128. Both intervals were unscored by the earlier experiments. Each has
8,128 targets; the nine models evaluate 146,304 checkpoint-target pairs over
**16,256 common unique text targets**, not nine independent corpora. The source
must beat both independently trained controls and its own memory-off/head-mean
ablations by at least 1% relative PPL with a positive lower paired-window
excess-NLL bound. Every final seed and both splits are retained.

| Seed | Full-memory test PPL | Trained head-constant PPL | Trained memory-off PPL | Full PPL excess vs constants, paired-window 95% interval | Qualified on calibration / test |
| --- | ---: | ---: | ---: | ---: | --- |
| 3 | 65.3118 | 64.5886 | 64.5235 | +1.120% [+0.793%, +1.455%] | No / No |
| 7 | 64.7593 | 64.3584 | 64.2884 | +0.623% [+0.364%, +0.882%] | No / No |
| 11 | 65.0777 | 64.5921 | 64.5268 | +0.752% [+0.462%, +1.050%] | No / No |

Full memory is also **0.732–1.222% worse than trained memory-off** on test;
all paired-window intervals support the cheaper control. A separate unadapted
Qwen diagnostic on identical model/data hashes has test PPL **93.9306**.
Late-decoder adaptation therefore improves this corpus substantially, while
adding this memory table does not improve on the cheaper adapted models.
The unadapted reference is post-run context and selects no model or gate.

There is a smaller, distinct positive result: replacing rows by their mean
**within the full trained checkpoint** raises test PPL by **0.566–0.759%**, with
positive paired-window lower bounds. Row identity now carries some signal in
that checkpoint. This does not imply that the large table outperforms a model
trained with constants from the start: the independent control does better.
Disabling the full model's branch itself lowers test PPL on every seed.
Both observations are retained rather than treating a positive ablation alone
as evidence of deployment necessity.

Independent-control intervals use 10,000 paired-window resamples, Python
`random.Random(731)`, with linear quantiles. Same-checkpoint ablations retain
the existing NumPy bootstrap, seed 731 / 10,000 resamples. Text dependence,
three training seeds and the chosen late-layer setting limit population or
architecture-level inference. No codec is selected or rerun on these
unqualified models.

## Preservation, storage and runtime boundaries

Training reuses exact FP32 frozen prefixes; only the two decoder layers and
the selected branch parameters can change. Full/cached logits and **all
trainable gradients** match before and after training, for every control.
Prefix/head hashes remain unchanged. Independent controls verify protocol,
initial memory/decoder hashes, split/sample hashes and serialized prefix bytes
before reuse. Prefix caching avoids redundant training work; it is not a
serving-memory measurement.

The six memory models export every row and original head offset. Native
matched-schedule evaluation has zero changed logit bytes, zero KL and zero
changed predictions in both splits. First-window complete forwards independently
check the suffix schedule. Full-memory tables change 112,740–112,803 allocated
rows during training; the observed failure is not simply an untouched table.

Native full-memory directories occupy **13.9738–13.9746 MiB**, saving only about
0.21% of raw rows. Complete bundles are **2,416,405,334–2,416,406,212 bytes**;
the backbone dominates and complete-file savings are below 0.001%. The trained
constant vectors have 768 bytes of payload, excluding branch/head metadata;
the compatibility checkpoint expands them to the original logical rows before
export. Its native memory directory occupies 1.2439–1.2478 MiB. That expansion
is a compatibility artifact, not a necessary or optimized constant deployment.
A directly indexed six-vector constant implementation would not require a
large n-gram table. Ordinary trained off checkpoints occupy 2,400,118,681 bytes.
[Inventory](inventory.json) includes every checkpoint/bundle file size and SHA.

Memory-off saves an ordinary Qwen checkpoint with the unused branch removed;
reload reproduces complete-forward logits byte for byte on the first window
of each split. All nine saved tokenizers reproduce the original tokenizer's
complete split token-ID hashes. Original pinned-tokenizer regex/length warnings
remain in the logs; no tokenizer patch is silently applied.

Training kernels take roughly 50–54 seconds per model on these devices, while
export, hashing and quality evaluation add work. These durations are **not
isolated serving benchmarks**. All quality workers retain dense references
and CPU prefix caches. No new throughput or deployment-memory improvement is
claimed from their wall time or allocator behavior. Existing warm-row-cache
serving results remain in the [separate experiment](../qwen_gpu_grid_evidence_20261106/README.md).

## Reproduce and audit

```bash
python -m tools.train_tinyengram_joint --device cuda --seed 3 --control full \
  --base-model /path/to/Qwen3-0.6B --corpus /path/to/TinyShakespeare.txt \
  --upstream-dir /path/to/pinned/TinyEngram --native-library /path/to/libmeft_rows.so \
  --protocol docs/qwen_joint_gpu_protocol_20261106.json --output-dir /new/full-seed3

# Other controls reuse that seed's audited frozen prefixes.
python -m tools.train_tinyengram_joint --device cuda --seed 3 --control head-constant \
  --base-model /path/to/Qwen3-0.6B --corpus /path/to/TinyShakespeare.txt \
  --upstream-dir /path/to/pinned/TinyEngram --native-library /path/to/libmeft_rows.so \
  --protocol docs/qwen_joint_gpu_protocol_20261106.json \
  --prefix-from /new/full-seed3 --output-dir /new/head-constant-seed3

python -S docs/qwen_joint_evidence_20261106/verify.py
```

The [launcher](run.sh) records all nine exact commands and shard assignments;
[raw reports](training/), [summary](summary.json), [runtime](runtime/),
[environment](environment.json) and [manifest](manifest.json) preserve
provenance without distributing corpus text, vocabulary, token IDs or weights.
The dependency-free audit reproduces independent-control bootstrap intervals,
point PPL, pairing, fixed qualification decisions and complete file accounting.
The saved NumPy intervals for same-checkpoint ablations can be regenerated from
per-target records with `runtime/lm_quality_metrics.py` and NumPy.

The [26-test pinned CPU/CUDA regression](final-pinned-tests.log) passes with one
MPS skip, including all joint-training controls, gradient equality, saved
checkpoints and original/native inference. [Local tests](local-tests.log) pass
231 cases with 120 optional-dependency skips, reported separately.

Hardware is 3 of 8 A100-SXM4-80GBs, AMD EPYC 7742, Linux, Python 3.11.15,
Torch 2.11.0+cu130, Transformers 4.57.3, NumPy 2.2.6. FP32 deterministic
arithmetic, TF32 disabled, no autocast, eight training threads and one interop
thread. Inputs are cached locally, with no download or provider charge.
Only hash/argv/cwd-verified matmul occupancy processes on GPUs 5/6/7 were
released under the user's authorization; unknown jobs and other GPUs were
preserved. Training launch gates on no competing compute client on these three
devices; CPU/I/O and the rest of the host are shared. The task release helper
ends with the experiment; [cleanup](cleanup.json) records that both the launcher
and release helper have exited. Auxiliary correctness checks allow shared GPU use
and make no isolated timing claim.

The next LLM capacity/codec study should start from an independently useful
trained memory whose rows beat trained simpler controls on held-out data.
This experiment narrows that requirement and keeps native fidelity intact;
it does not supply that missing checkpoint.
