# Real Qwen identity memory: native costs and fidelity

The source-checkpoint → native bundle → independently reopened Qwen path now
works on six **trained 20,877-edit exact-key checkpoints** from the
[independent-row experiment](../ngram_identity_evidence_20261107/README.md).
No model was retrained. This is frozen-backbone counterfactual editing, not LLM
pretraining, generated-answer evaluation or a billion-row table.

The practical result is original-ID, FP32-byte-preserving GPU offload with similar
observed complete-prefix latency. **These tables do not demonstrate a lossless
disk-compression or DATrie storage advantage.** All physical rows, including
sentinels and 70/80 never-addressed budget-padding rows, remain present.

## Complete model comparison

Each model has seeds 3/7/11 and 64 chronological official test prefixes from
rewrite, paraphrase and neighborhood groups. Workers independently load the
model, verify frozen weights, run complete uncached forwards, warm once and repeat
three times. Source/XOR gives 384 checkpoint-prefix pairs; DATrie layout at seed 3
adds 128 comparisons. All **512 paired next-token logit hashes are byte identical**.
These are related repeated prefixes, not 512 independent quality observations.
All original row bytes are also checked. The separate RSS audit adds six paired
model-prefix checks without changing the main comparison count.

| Model | Original row payload | CUDA allocated reduction after open | Fraction of model's allocated CUDA bytes | Source / native p50 across seeds |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 74.356 MB | 70.913 MiB | 2.411% | 45.497 / 45.789 ms |
| Qwen3.5-4B | 73.514 MB | 70.110 MiB | 0.403% | 78.528 / 78.111 ms |

Latency numbers are medians of three per-seed p50s. Native/source ratios range
0.9926–1.0158 for 0.6B and 0.9872–1.0027 for 4B; small differences do not establish
a speed advantage. Measurements use A100 80GB, FP32, eager attention, batch one,
no KV cache or padding. Qwen3.5 logged a missing optional fast path and used the
Torch fallback in both variants. These are not optimized production-serving
timings. Selected-GPU occupancy is checked before context creation and
before/after timing; shared-host CPU/I/O competition remains possible. Output
readback/hashing is outside timed forwards.

Full startup peak RSS remains dominated by base loading and fingerprinting:
about 5.750 GB for 0.6B and 28.547 GB for 4B in the separate seed-3 audit, with
essentially equal source/native peaks. Avoiding the temporary dense memory matrix
does **not** establish bounded whole-model startup memory.

## Storage and algorithm ablation

Serving metadata counts. Native bundles include memory files, unchanged nonrow
delta, original training vocabulary and model recipe. The shared base remains
separately required and its directory bytes are recorded. These are seed-3 values;
all seeds and unfavorable results are in [summary.json](summary.json).

| Model | Source delta/vocabulary/recipe bundle | Native XOR bundle | XOR table/metadata | DATrie layout table/metadata/index | Registered-key exact structural merges |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B | 77.415 MB | 77.976 MB | 74.918 MB | 123.437 MB | 0 |
| 4B | 78.987 MB | 79.597 MB | 74.124 MB | 122.865 MB | 0 |

The native default uses 32-row blocks. Identity, XOR and generic exact dedup have
similar costs. Generic dedup can merge sentinels; it does not establish shared
learned-key vectors. DATrie layout/exact mark-compact use the production planner
and persisted original-ID index, verifying every registered key and original row.
Exact structural merging is zero in all six checkpoints. The added index costs
48.5/48.7 MB with this fixed-width canonical-token string encoding. The original
Python dictionary still addresses Qwen memory: DATrie is an additional
index/layout ablation, not a dictionary replacement.

Whole-table gzip level 6 saves about 7.1% losslessly, but has no independent
random row access. A disclosed post-baseline diagnostic uses 32/128/512/2048-row
blocks at seed 3. Best native row-payload factors are only 1.009/1.008; first-trace
p50 grows from about 0.52 ms to 20–21 ms at 2048 rows. Warm p50 remains around
0.064–0.066 ms because the repeated trace benefits from row caching. This is an
empty application-cache trace with warm OS cache, not cold disk latency.

Seed-3 dense / raw-mmap / native-XOR CPU row-query p50s are
0.0425 / 0.0544 / 0.0662 ms for 0.6B and 0.0430 / 0.0553 / 0.0639 ms for 4B.
The allocating tensor-gather API is timed; token normalization, key lookup, GPU
and model work are excluded. There is no native CPU speed advantage here.

The format stores an 8-byte forwarding entry per original row even for an
identity map: 6.25% of a 32-dimensional FP32 row. Removing that redundant map
is a concrete next optimization. Compact key encoding and replacing the Python
dictionary are separate opportunities; merely constructing DATrie does not
deliver them.

## Anonymous and file-backed RSS

The separate seed-3 audit records Linux RssAnon/RssFile/RssShmem for four CPU
readers and source/native models, with 64 row prefixes and three model prefixes.
Its timings are not used above. Process-wide RSS includes imports, allocator
retention, base loading and outputs; CUDA allocator bytes, anonymous/file-backed
RSS and disk bytes are different costs.

CPU reader anonymous RSS growth after open is roughly 76 MB for dense versus
5 MB for native or raw mmap. After querying, native file-backed RSS growth is
about 9 MB versus 4 MB for raw mmap. Whole-model anonymous RSS after open falls
by about 54/60 MB, but startup peaks remain essentially unchanged. This single
process audit establishes no universal RSS reduction or mmap working-set bound.

## Reproduce and inspect

```bash
python3 -S docs/qwen_native_identity_evidence_20261109/verify.py
python -m tools.benchmark_counterfact_storage prepare --help
python -m tools.benchmark_counterfact_storage worker --help
python -m tools.benchmark_counterfact_storage sweep --help
python -m tools.benchmark_counterfact_model --help
```

The exporter reads the trained row/nonrow delta and vocabulary without loading
a dense Qwen backbone. Bundles work with `counterfact_native.reopen`; native
query workers never open original dense rows. Reports retain exact commands,
environment, source fingerprints, request hashes, samples and memory snapshots.
Weights, token keys, models and query arrays remain in the original directories;
this archive publishes costs and hashes, not those payloads.

[verify.py](verify.py) checks archive integrity, previous trained-run provenance,
paired hashes and arithmetic using only the standard library. Rechecking row
bytes/logits requires checkpoints and benchmark commands. Runtime snapshots are
grouped by recorded full commit. In these reports `environment.source_sha256`
refers to the shared storage-report helper; both benchmark modules are archived
without relabeling that digest. Compiler/filesystem/OS/GPU details and cost
assumptions are recorded.

The first GPU attempt was blocked by a new occupancy program; failed reports
remain in `failures-attempt1/`. The retry passed GPU gates. No training/evaluation
process was stopped. Finite benchmarks and occupancy guards finished after the
measurements. Existing shared hardware was used; monetary/energy costs were not
measured.

This supports the frozen-memory integration and identifies its next engineering
cost. It adds no evidence for natural co-training cosine above 99%, lossless
structural compression of these rows, full-model pretraining or superiority over
optimized serving/quantization methods.
