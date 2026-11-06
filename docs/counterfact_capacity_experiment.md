# MultiCounterFact memory capacity experiment

This experiment tests whether learned n-gram rows help batch knowledge editing
when edit count and hash capacity change. It does not establish a state-of-the-art
editing algorithm or a need for a storage backend. Those require independent
method comparisons and source-qualified serving measurements respectively.

The [completed 1,024-edit three-seed grid](counterfact_small_evidence_20261107/README.md)
does not show a stable large-table gain; BM25 has the highest observed quality,
and both native exports slightly expand. Its 40 main evaluations, four separately
registered frozen-backbone probes and two byte-exact native reopen checks are
archived with a scalar verifier. The 20,877-edit grid is still running and is
not included in those completed results.

The [frozen protocol](../benchmarks/counterfact_capacity_v1.json) uses Qwen3-0.6B
and Qwen3.5-4B, 1,024 and all 20,877 official MEMIT MultiCounterFact edits, three
training seeds, three hash capacities, trained memory-off, trained head constants,
and a parameter-budget-matched dense adapter. Train the last eight original text
decoder layers with each control under the same schedule. An unedited base and
BM25 retrieval from edited rewrite/new facts provide additional controls. Retrieval
uses postings, not a deliberately slow full-table scan; top-1/top-3 selection uses
the calibration fold only.

Every official rewrite, paraphrase and neighborhood is retained. Score the new
and old target with the original MEMIT NLL comparison, complete-target
teacher-forced accuracy and the three-way harmonic score. Generation entropy and
TF-IDF consistency are **not measured** by this protocol. MultiCounterFact is
counterfactual knowledge updating, not a test of independently verified current
facts. The edited rewrite facts train in both folds, as in knowledge editing;
paraphrases and neighborhoods never train. Shared neighborhood prompts can be
edits elsewhere in the batch. Keep the official metric and flag those prompts
for a separate diagnostic rather than removing them.

Use a separate environment from the old TinyEngram integration:

```bash
python -m pip install -e '.[counterfact-research]'
python -m unittest tests.test_counterfact_data tests.test_counterfact_retrieval tests.test_late_memory_replay -v
```

Get `multi_counterfact.json` from the
[official dataset endpoint](https://rome.baulab.info/data/dsets/multi_counterfact.json)
and `experiments/py/eval_utils_counterfact.py` from
[MEMIT commit 80426fd](https://github.com/kmeng01/memit/blob/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/experiments/py/eval_utils_counterfact.py).
The loaders enforce SHA256 for both sources. The Engram implementation is pinned
to [TinyEngram commit 317f238](https://github.com/AutoArk/TinyEngram/tree/317f23832aa8f8e382a0916ec469e81ba12b6dff).
Keep each upstream license with its local source snapshot.

Run `python -m tools.counterfact_preflight --help` first, then
`python -m tools.counterfact_capacity --help` for preparation, cache shards,
training and baseline evaluation. The finite remote queue in
`tools/counterfact_runner.py` is specific to the user's known machine and task
directory; it is not a portable GPU discovery tool and never terminates other
users' processes. The independently audited occupancy helper is task-specific
and is stopped by the queue on completion or failure.

For an already stopped queue, `counterfact_runner --resume` preserves its prior
outcomes and hash-validates completed runs/cache shards before reusing them.
It checks shard sequence coverage against prepared batches and rejects partial
or changed files. Preserve partial artifacts separately before a retry. Ensure
the previous helper/children stopped, preserve logs, and clear task-local
completion markers before restarting the finite queue.

Frozen-prefix caching uses fixed batch size eight and exact length buckets.
Duplicate fillers have no target loss and do not count as training exposures or
metrics. Check cached prefixes, complete logits and every masked-loss gradient
before training. The original decoder modules handle rotary positions, causal
masks and Qwen3.5's mixed linear/full attention; there is no custom reconstruction
of the backbone. Final deltas plus the immutable base form the deployment recipe.

FP32 parameters and PyTorch matmul/cuDNN TF32 flags do not force third-party
Triton kernels to use IEEE arithmetic. This remote installation's FLA triangular
solver explicitly uses TF32. Preserve that implementation and record its source
hash and precision audit. On the four-case Qwen3.5-4B preflight, padded official
batch scoring versus unpadded scoring differs by up to 0.0013633 NLL; target
accuracy and answer ordering agree. Given the same logits, the unchanged official
formula agrees exactly. This small check is not proof that batch arithmetic can
never change a near tie over the full benchmark. Report near ties and the runtime
path with final results; cache equivalence checks use the same batch shape.

Run artifacts must include the source commit, protocol/data/tokenizer/weight
hashes, exact command, environment and GPU, original per-candidate scalar NLL and
accuracy, all per-case metrics, complete model/delta costs, and failed-run logs.
Prefix states, token IDs and model deltas stay in the private run directory.
Do not summarize an incomplete queue as completed evidence. A full table losing
to trained cheaper controls is a useful negative result, not a capacity win.
