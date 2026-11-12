# Independent-corpus and fused-query protocol

Saved before new training, fitting or query measurements on 2026-11-02.
The previous TinyShakespeare results motivate this replication but do not
select the new corpus, seeds, codecs or quality budgets using new test scores.

## Independent data and matched training

Use the cached enwik8 zip from <https://mattmahoney.net/dc/enwik8.zip>, SHA256
`547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc`.
Read the first 4 MiB of member `enwik8`, tokenize with the unchanged lowercase
English regex, and retain exactly the first 200,000 tokens. Do not strip XML or
wikitext: the result includes metadata/markup tokens and is a benchmark prefix,
not a cleaned prose corpus. Preserve raw-prefix, zip and tokenized-file hashes.
Use contiguous 80/10/10 token splits. Fit the 1,200-word vocabulary, count>=5,
and complete n-gram incidence/multiplicity signatures on training tokens only.
This is distinct from TinyShakespeare but not independent of the earlier
enwik8 capacity study; do not describe enwik8 as never previously examined.

Maximum orders 4/8/12; seeds 3/7/11; 128 dimensions; 2,000 SGD steps of 256
sampled context pairs, LR 0.05, five frequency^0.5 negatives. Pair each dense
model with a group-mean model on identical initialization, samples and
negatives. Run both legacy average-forward/unscaled-input-gradient and standard
average-derivative conventions: 18 pairs total. Record actual initial weights,
update directions, mean trajectories and center coverage. Report held-out
SGNS objectives and context discrimination using 8,192 paired samples per split,
fixed seed 42/43, plus vocabulary coverage. The group-size gradient correction
is fixed; no hyperparameter, convergence or semantic-quality claim follows.

## Complete fixed-quality codec comparison

Use all nine **legacy-convention** new dense checkpoints for the primary
comparison, matching the earlier FAISS study's convention. Do not combine the
average-derivative checkpoints into its headline. Fit the complete previously
declared 19-config grid: FP32, FP16, SQ8/6/4, full-vector clusters with
256/512/1024/2048 centers, PQ8x4/8x8/16x8/32x8/64x8,
OPQ8x8/32x8/64x8 and RQ2x8/4x8. Compare original-ID direct codecs,
certified groups and full-domain hybrid, 513 base candidates. Audit every direct
encoded-byte dedup alternative (171); include beneficial alternatives in
validation selection. Fit at most 16,384 identical sampled original row IDs,
with occupancy-weighted group means, seed 42, 20 codec/10 OPQ iterations,
four fit threads and one query thread.

Select smallest complete bytes on validation under three predeclared SGNS
loss-change budgets: 0.01%, **0.1% primary**, 0.5%; discrimination decline at
most 0.5 percentage point and known-word aggregate relative L2 at most 5%.
Audit test afterward; preserve failed/ineligible candidates and paired
token-block intervals. Include word/output rows, key registry, codes,
codebooks, maps/certificates and every original nonsingleton fallback row in
complete costs. Report fixed-trace plans separately. Count known-word
precomputation as the stronger restricted inference control.

Use 512 lexical sampled known/OOV test bags where available, two warmups and
15 repeats in fresh query processes. Report replay and compilation-plus-query,
nonempty/rejected coverage and all failure logs. These settings replicate a
predeclared finite grid, not all quantization algorithms or tuning budgets.

## Fused original-ID counting

Share one exact original-ID occurrence count between certification and weighted
schedule construction. Apply batch counting to direct and encoded-byte-map
controls too. For unsafe hybrid bags, keep the original full fallback and its
ordering. For the production native plan, retain the same representative IDs,
float32 weights, offsets and mean denominators.

Pin the reference FAISS and native plan modules to main
`aa38f3371a1a8ac8e35dd258cb955688f6329fa0`. On the previous nine frozen
TinyShakespeare checkpoints and validation-selected codecs, require all saved
schedule arrays and all registered-word outputs to be byte-identical. Use fresh
processes and paired interleaved old/new compilation, replay and total-query
timings. Repeat the same comparison on the new corpus selections after fitting.
Keep the reducer and codec files unchanged. Native plan construction may be
measured separately; do not infer native serving acceleration from a FAISS
adapter or from compilation alone.

All work uses CPU under `/llm-align/duyimin/ngram-cross-corpus-20261102`,
`CUDA_VISIBLE_DEVICES=''`, no GPU process inspection/stopping. Shared warm
CPU timings are not exclusive/cold-page/production LLM measurements. Record
source SHA, exact commands, source/corpus/model hashes, OS/CPU, package versions,
threads, raw samples and limitations. Publish metrics/hashes, not corpus or
checkpoints. Monetary/energy cost and bounded construction RAM are not measured.

Operational follow-up before dispatching remaining checkpoints: execute at
most three independent checkpoint jobs, with four fit threads per job and one
thread in each fresh query worker. Reuse only complete serial checkpoint runs
with identical corpus/checkpoint hashes, engine source hashes and all declared
settings; preserve their original commands/source SHA. The controller records
each phase and verifies the complete 57-variant grid for every checkpoint.
This changes elapsed-time orchestration after initial checkpoints, not fitting,
the grid, quality gates or validation selection. Query timings remain shared-CPU
measurements. Preserve the interrupted serial continuation and its log.
