# Real Qwen interfaces for official MultiCounterFact — 2026-11-06

**This archive validates interfaces and scoring; the formal training grid is
running and is not completed evidence.** No memory-capacity or storage advantage
is claimed from this preflight.

Source commits: `9c35e8f8c3eca2967d6ba50d4a7e6f7576a6f55c` for the 4B preflight;
`537debf3e8d9715afc464a9e6db8e97e34fb0bfe` for the final 0.6B preflight and finite
training queue. Commands and source-file hashes are embedded in each report.
The [experiment protocol and reproduction guide](../counterfact_capacity_experiment.md)
describe the independent controls and interpretation limits.

| Validation | Qwen3-0.6B | Qwen3.5-4B |
| --- | --- | --- |
| Actual local vendor weights load | No missing, mismatched or unexpected keys | No missing, mismatched or unexpected keys |
| Full/cached FP32 logits, same batch | Byte-exact | Byte-exact |
| All masked-loss parameter gradients | 97 groups exactly equal | 115 groups exactly equal |
| Same logits, unchanged MEMIT formula | Exact agreement, 104 choices | Exact agreement, 104 choices |
| Padded official batch vs unpadded NLL | Max difference 0.0000486374 | Max difference 0.0013632774 |
| Ordering / target-accuracy changes in those choices | 0 / 0 | 0 / 0 |

Seven CPU/CUDA fixtures passed across real Qwen decoder classes, cropped logits,
memory enabled/disabled, and an optimizer update. The initial CPU fixture error
is retained: installed CUDA-only FLA was incorrectly dispatched on CPU. The
CPU test now explicitly chooses the upstream torch fallback. The remote CUDA
model retains the installed stock FLA path. Its triangular solver uses TF32
despite the PyTorch matmul flags; [precision audit](precision-audit.json) records
the exact source hash. Small preflight ranking agreement does not guarantee
that all full-benchmark near ties are insensitive to batch arithmetic.

The first 4B scoring preflight rejected an absolute NLL tolerance of 0.0001
between different batch shapes. The replacement check tests the unchanged
official formula exactly on identical captured logits, and separately records
all cross-batch NLL drift, ordering and accuracy changes. The implementation
change is inspectable in commit `9c35e8f`; the first 4B host log was overwritten
by the rerun and is not represented here as an archived raw failure log.

The official data bytes are fixed to
`977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833`.
Both tokenizers accept all 20,877 edits and every official paraphrase/neighborhood
without a prefix/suffix boundary mismatch. Preparation covers more than 374k
unique sequences and 4.3M tokens, and flags 4,189 neighbor prompts that also occur
as another edited rewrite. These counts are preparation evidence, not accuracy
measurements. Token IDs, prefix states and model deltas remain remote under the
user's task directory. The [model manifest](model-manifest.json) identifies the
read-only weight/tokenizer inputs.

```bash
python -S docs/counterfact_capacity_evidence_20261106/verify.py
```
