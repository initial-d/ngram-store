# Prefix/suffix node sharing: complete quality, capacity and query evidence

This study connects prefix/suffix tree-node sharing to actual held-out subword
context quality and complete serving costs. It also tests why standard
average-derivative training initially produced less `.99/.99` compaction than
the legacy training rule. Related nodes mean parent–child or multi-level paths
in a prefix trie or a trie built from reversed strings; relatedness alone is
not a similarity or quality guarantee.

The [fixed protocol](../tree_node_quality_protocol_20261109.md) records all
choices before these measurements. The original 200k-token enwik8 prefix had
already been used for earlier research; this follow-up is exploratory there.
A new token interval `[200000,300000)` provides a held-out confirmation segment
for these models, from the same corpus and preprocessing. It is not an
independent corpus or a claim of globally untouched data.

## Completed evidence

- All 18 existing trained checkpoints: two gradient conventions, maximum
  orders 4/8/12, seeds 3/7/11. Each gets all 204 declared candidates, including
  raw and native precision controls, both joint tree planners, the complete
  19-codec FAISS grid, every codec's encoded-byte dedup alternative, dense
  serving and three constant-row necessity controls.
- Both ordinary inference and source-aggregate precomputation for all known
  words. OOV strings still use the candidate n-gram backend. All adjacent-window
  context pairs with registered output targets are scored, including OOV input
  centers and empty stored-ngram bags; original positions remain intact.
- Validation-only choices at five loss budgets, with constraints on both
  all-center and nonempty-OOV loss and discrimination. Original test and the
  new confirmation segment audit frozen names. Every candidate has paired
  128-token-block sufficient statistics; every unique selected name has
  2,000-resample pointwise intervals.
- Fresh query workers for all candidates and both inference modes. The final
  selected/control retiming runs sequentially after this task's builders,
  trainers and other evaluators exit. DATrie and independent hash indexes are
  measured; native candidates get matched 0/1 MiB row-cache controls.
- An additional complete row audit reopens every bundle and reports final
  post-encoding cosine, norm ratio and relative-L2 errors. Original-vector
  merge gates do not silently become guarantees after extra quantization.
- 36 fixed-duration training trajectories and all 108 checkpoint analyses.
  All 18 scale-1 original 2,000-step input/output tables replay byte for byte
  before extension. Both n-gram initialization scales and all durations remain
  in the report, including unfavorable quality and capacity observations.

## Main findings

The mechanism hypothesis has support under standard average derivatives.
At maximum order 12, scale-1 n-gram initialization and `.99/.99` C++ DATrie
sharing, median native **n-gram-only** saving across three seeds is 1.34% at
2,000 SGD steps, 12.08% at 8,000 and 41.49% at 32,000. Accumulated-update
norm relative to initialization grows from 10.59 to 16.94 to 32.77; common
update dispersion within incidence groups remains near `1e-7` relative.
The matched `.1` initialization condition and absolute source task quality
are also retained. This supports a conditional training mechanism rather than
a universal final-cosine rule or an instruction to shrink initialization.
Source confirmation loss rises between 8,000 and 32,000 steps in every
order/convention/initialization group. Increased compressibility therefore
does not demonstrate improved generalization or justify longer training as a
quality optimization; the fixed-duration experiment isolates a mechanism.

The complete initial-checkpoint comparison limits the backend claim. At the
primary 0.1% validation loss budget, the selected FAISS solution, including its
byte-deduplicated alternatives, is smaller in every order/convention/inference
mode than either selected native joint tree planner. Native tree sharing does
not establish universal capacity/quality/query superiority here. Longer-training
mechanism points evaluate `.99/.99` native sharing; they do not repeat the full
FAISS frontier and cannot establish a new Pareto win against FAISS.

Query cost also favors the selected FAISS solutions on these initial models.
In ordinary inference at the primary budget, across all 18 models, median
complete query p50 relative to matched raw FP32 is 2.88 for C++ DATrie sharing
on known words and 2.84 on OOV strings. Direct FAISS ratios are 1.02 and 1.02;
FAISS with encoded-byte dedup gives 1.02 and 1.04. These are validation-selected
solutions, including the held-out budget failures reported below, on warm
shared CPUs. The [capacity/query plot](figures/capacity-query.png) marks those
failures; `summary.json` also retains every matched hash/DATrie and row-cache
control, load time and RSS. Neither native speed superiority nor a whole-model
41.49% saving follows from the mechanism result.

Frozen primary choices are audited without reselecting on either segment:

| Family | Selected model/mode choices | Original-test point failures | Confirmation point failures |
| --- | ---: | ---: | ---: |
| Raw precision | 36 | 0 | 0 |
| Dense FP32 | 36 | 0 | 0 |
| Native identity | 36 | 0 | 0 |
| C++ DATrie joint sharing | 36 | 0 | 0 |
| Independent tree reference | 36 | 0 | 0 |
| Direct FAISS | 36 | 7 | 0 |
| FAISS + encoded-byte dedup | 36 | 7 | 0 |
| Constant-row family | 9 | 8 | 0 |

Direct/deduplicated FAISS failures overlap and are not independent replications.
These are point-estimate budget tests on all-center and nonempty-OOV loss and
discrimination; the archived paired intervals remain separate. Confirmation
passing does not erase an original-test failure or establish generalization.

The training-occurrence-weighted constant row qualifies on validation and
confirmation in all nine average-derivative models **with known-word source
aggregates precomputed**. Eight fail the original-test budget. All three
constant controls fail primary validation in ordinary inference. Thus the
necessity of distinct stored n-gram rows depends on the workload, inference
contract and evaluation segment. Full addressing alone does not prove useful
memory is indispensable after stronger frozen-inference alternatives.

[Training-duration capacity](figures/mechanism-capacity.png),
[update scale and source quality](figures/mechanism-update-quality.png),
[ordinary capacity frontier](figures/capacity-ordinary.png),
[precomputed-word frontier](figures/capacity-cached.png),
[cosine × norm gate grid](figures/joint-gates-datrie.png), and the
[complete numeric summary](summary.json) keep these tradeoffs inspectable.

## What the quantities mean

SGNS loss and context discrimination assess the frozen finite subword context
model. Negatives are sampled with frequency exponent .5, five per pair; target
collisions are retained. This is not classification accuracy, semantic task
quality, full-softmax perplexity, Engram pretraining or LLM utility.

Complete query timings cover string n-gram generation, key lookup, original-ID
count/schedule construction, decode and weighted aggregate. Replay alone is
reported separately. They do not include the SGNS output scorer or an LLM
request. All measurements use shared CPUs and warm filesystem/page-cache
conditions, rather than exclusive hardware, disk-cold serving or production
latency. Raw samples, p50/p95, load, RSS, construction time and peak memory are
retained. Lower bytes alone do not establish faster or better serving.

The strongest frozen inference control precomputes the original full aggregate
for every registered word, replacing the same-size word-row file. It keeps
OOV serving, and its known-word precision is exact to the common source
reducer. Its success is not evidence that arbitrary n-gram row lookup is
needed. Constant rows preserve the original denominator, so their effects
cannot be attributed to silently changing normalization.

## Capacity accounting correction

The original frozen candidate ledger counted row bundles, codes/codebooks,
scales, maps, DATrie keys, word registry, word and output rows, but omitted the
common `metadata.json` that the measured reader opens. `complete-cost.json`
adds the actual private file's bytes to every candidate in that model.
`complete_serving_bytes` is the corrected headline quantity. Original build
reports, their hashes and the frozen selection are preserved. This constant
addition cannot change a minimum-byte selected name; it does change the exact
reported capacity ratios. Hash-index costs replace DATrie index bytes; the
known-word cache replaces an equal-size word-row file. Runtime binaries,
Python/library dependencies and evaluation inputs are outside model-file costs.

N-gram-only native saving in the mechanism experiment excludes the common
word/output/index files and must not be quoted as whole-model saving. Its
8192-byte block budget also differs from some earlier reports; compare the
matched controls here rather than mixing historical format configurations.

## Reproduction and audit

```bash
python -S docs/tree_node_quality_evidence_20261109/fetch_raw.py
python -S docs/tree_node_quality_evidence_20261109/verify.py
python docs/tree_node_quality_evidence_20261109/summarize.py
python docs/tree_node_quality_evidence_20261109/plot.py
```

The fetch command verifies every archive and member against the pinned
`raw-manifest.json`, and stores the complete numeric attachments under
`artifacts/tree-node-quality-20261109/raw`. Existing verified members are reused.
The audit command uses only Python's standard library. It checks the manifest,
complete grid and source coverage, original checkpoint hashes, serving-byte
sums, immutable validation choices, paired-block arithmetic, timing statistics,
bootstrap sample quantiles, 108 factorial points and all 18 logged exact
replays and the full-domain input/cache integrity report. It cannot independently recreate unpublished weights, native
reconstruction assertions, lexical queries, training or cosine measurements.
The NumPy bootstrap draws are archived; the stdlib audit verifies interval
arithmetic without regenerating those draws. Intervals are pointwise within-text
approximations, not independent-corpus or simultaneous guarantees.

`runtime/` preserves each actual measurement revision. `auxiliary/` records
hardware/compiler details, source-export/build commands and the initial failed
build before training. Environment paths in reports and scripts use explicit
`${RUN_ROOT}`, `${TRAINING_ROOT}`, `${ENV_ROOT}` and `${CORPUS_ZIP}` placeholders.
Numeric values and raw private report hashes are retained. Source tables,
weights, lexical keys and corpus files are not published.
Historical source archives also replace user workspace and model-root paths
with `${USER_WORK_ROOT}` and `${OPEN_MODELS_ROOT}`. The original archive and
affected-file hashes are retained in `auxiliary/source-path-redactions.json`;
the algorithm files used by source-provenance checks remain unchanged. Adapt
these path placeholders to your workspace when reproducing the archived phases.

To rerun measurements, reconstruct the original checkpoints using the
[independent-corpus reproduction](../cross_corpus_evidence_20261102/README.md),
then use the archived phase-specific source revisions and commands in
`auxiliary/`. Empty `CUDA_VISIBLE_DEVICES` is mandatory. Build native libraries
from the same snapshot with `make native structural-native`. No GPU was
inspected or modified in this CPU experiment. Existing hardware/assets were
used; energy and monetary cost were not metered.

## Failures and numerical corrections

The original generic environment helper writes `threads: 1` in every phase.
It is not an observation of FAISS build parallelism: every builder explicitly
sets FAISS to four threads, and queries set it to one.
`faiss-codec-metadata.json` preserves all 684 original codec metadata records,
including `training_threads: 4`, with hashes matching the frozen serving files.
The generic label, original reports and actual phase source remain unchanged.

The complete row audit initially failed on 108 OPQ direct/deduplicated
candidates. Four-thread construction and one-thread reverse transformation
produce different FP32 rounding. Each repair verifies every serialized serving
file, reproduces the construction decode byte for byte, then audits all actual
one-thread query rows and both validation modes. Across those 108 candidates,
maximum absolute decoded drift is `1.6689300537109375e-6` and maximum relative
Frobenius drift is `1.1920079102407778e-7`. None changes eligibility at any of
the five declared budgets. Frozen selections remain unchanged. Archived
one-thread validation paired blocks support the gate recheck; original failures
and logs are retained.

Final hash retiming exposed a separate loader omission: saved `keys.flat`
could not reopen for the already implemented hash lookup. The repair loads
validated key/ID offsets and builds a real owned `std::unordered_map`; sorted
binary search is a distinct policy. The initial interrupted controller, failed
logs, original execution schedule and valid completed DATrie reports are
preserved. Recovery verifies retained hashes and completes all other requests
sequentially, using the separately recorded repaired library. Runtime hash
allocations count in RSS, and serialized hash-index bytes replace DATrie bytes.
