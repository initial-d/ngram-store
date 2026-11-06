# Held-out Qwen memory: branch effect is not row-identity value

This follow-up qualifies the existing 32-step TinyEngram/Qwen3-0.6B checkpoint
before investing in a compression competition. It excludes the first 16,384
TinyShakespeare characters used by the earlier branch-training protocol and
scores **8,128 next-token targets** in 64 nonoverlapping 128-token windows:
2,032 calibration targets and 6,096 independent test targets. The suffix is held
out from branch training; overlap with Qwen pretraining is unknown. This is one
small trained branch, not a released DeepSeek Engram checkpoint.

The important control is **head-mean**: every row within an original hash head
is replaced by that head's unweighted mean over the actual trained FP32 rows.
It keeps projections, gates, convolution, backbone, token IDs and execution
schedule unchanged while removing n-gram row identity. Memory-off instead
bypasses the entire branch. Neither is an independently trained baseline.

| Control | Calibration PPL | Test PPL | Test PPL change vs source | Test top-1 changes |
| --- | ---: | ---: | ---: | ---: |
| Original trained FP32 rows | 98.3445 | 116.5271 | — | — |
| Memory off | 110.9912 | 129.5668 | +11.1903% | 1,500 / 6,096 |
| One constant mean per hash head | 98.2186 | 116.5755 | +0.0416% | 121 / 6,096 |
| Native FP32, same execution schedule | 98.3445 | 116.5271 | 0% | 0 / 6,096 |

The memory-off test PPL interval is **+8.98% to +13.61%**; the head-mean
interval is **−0.0887% to +0.1757%** (paired-window 95% bootstrap).
Head-mean still changes 121 top-1 predictions and has max test KL 0.10518; low
average PPL change is not byte preservation or a general equivalence guarantee.
Native preserves every full-window logit byte across all 64 windows, with zero
KL and no changed predictions. These checks use the same full-forward schedule,
not a comparison of full and cached arithmetic.

The qualification rule was committed before the run: on calibration, the
source must outperform **both** ablations by at least 1% relative PPL, and the
paired-window bootstrap lower bound for each ablation's excess NLL must be
positive. Test results independently check the same rule; they select no
checkpoint, codec, text or threshold. The bootstrap is descriptive for these
fixed windows: adjacent text and model-seed uncertainty limit population claims.

**Qualification fails on calibration and is not confirmed on test.** The whole
trained branch improves PPL on this suffix, while retaining individual n-gram
row identity has not shown an advantage over head constants. We therefore do
not run or select further lossy codecs on this checkpoint, and do not present
it as proof of necessary memory-table capacity or superiority to FAISS.
This distinguishes a useful branch-level adaptation from a demonstrated need
for distinct retrieved vectors; it does not identify the causal mechanism of
the branch's benefit.

The original memory is only **648 × 8 FP32 = 20,736 bytes**. Its four head means
have 128 bytes of FP32 vector payload, excluding head-layout metadata. This
arithmetic is not a deployed constant backend or complete-model size result.
The existing native table and embedding metadata occupy 26,003 bytes,
**25.40% more than the 20,736-byte raw FP32 row payload**; this is a small-table
overhead comparison, not complete-checkpoint accounting. The backbone dominates
storage; the quality oracle deliberately retains source
memory and runs all controls in one resident model. Its wall time and RSS must
not be used as codec throughput, latency or deployment-memory evidence.

This finding does not test the prefix/suffix co-training hypothesis: arbitrary
hash-head rows are not certified structural or co-training groups. It does show
why “memory-off hurts” alone cannot demonstrate that many distinct retrieved
rows need to be stored. A convincing LLM storage experiment needs a larger
trained checkpoint whose row identity contributes on held-out inputs, followed
by the same-source native, raw quantization and full FAISS controls with isolated
serving workers. The frozen protocol has no new GPU training or retraining.

## Reproduce and inspect

From runtime commit `ba6b8f3`, supply the original local checkpoint, bundle,
cached corpus, pinned TinyEngram source and compiled native library:

```bash
cc -std=c11 -O3 -Wall -Wextra -Werror -fPIC -shared \
  native/meft_rows.c -lz -o /path/to/libmeft_rows.so
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python -m tools.evaluate_tinyengram_quality \
  --checkpoint /path/to/checkpoint --bundle /path/to/bundle \
  --corpus /path/to/tinyshakespeare.txt --upstream-dir /path/to/TinyEngram \
  --native-library /path/to/libmeft_rows.so \
  --protocol docs/qwen_quality_protocol_20261106.json \
  --report-out /new/path/qualification.json

python -S docs/qwen_quality_evidence_20261106/verify.py
```

- [Frozen protocol](protocol.json), [per-target report](qualification.json.gz)
  and [full run log](qualification.log) retain the schedule, exact command,
  input/output fingerprints, PPL, KL tails, top-1 changes and bootstrap intervals.
  [Compact summary](summary.json) contains the same decisions and aggregates;
  [native build inventory](native-build-inventory.txt) records the compiler,
  native source and shared-library hashes. The run took 264.68 seconds, with
  source and all quality controls sharing one process; this is not a timing contest.
- [Runtime source snapshots](runtime/) and [file manifest](manifest.json) make
  the reported implementation independently inspectable without redistributing
  model weights, token IDs, vocabulary or corpus text.
- [Remote CPU integration tests](integration-tests.log) exercise the evaluator
  on a trained miniature real-Qwen fixture; 19 tests ran, three accelerator-only
  tests skipped. Fixture training is an interface test, not application evidence.
- [Targeted tests](targeted-tests.log) pass all 23 checks. The local one-process
  [OpenMP failure](full-tests-openmp-failure.log) is retained; all 33 test modules
  pass when run in [separate processes](isolated-tests/summary.json), with normal
  optional-dependency skips. Linux/macOS CI checks remain separate.

Environment: AMD EPYC 7742, Linux x86-64, Python 3.11.15, PyTorch 2.11.0+cu130,
Transformers 4.57.3, eight CPU threads and one interop thread. CUDA devices were
hidden for the task. No external data/model download, GPU work or usage fee was
involved. The saved original tokenizer emits an upstream regex warning; it was
kept unchanged to preserve original token/hash addressing. The tokenizer also
warns about the complete suffix length; model calls each receive only 128
tokens. These warnings remain in the raw log.
