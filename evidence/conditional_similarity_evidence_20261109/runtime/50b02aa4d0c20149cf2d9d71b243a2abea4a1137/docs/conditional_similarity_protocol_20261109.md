# Conditional n-gram similarity: fixed analysis protocol

This is a post-hoc analysis of existing trained checkpoints. Earlier aggregate
similarity and application results are known; this specification is fixed before
computing the new frequency/length strata and matched-control results.

## Questions

1. Do some structurally related learned rows reach cosine >=0.99, and what
   fraction of rows/pairs and training occurrence mass do they cover?
2. Within a fixed length, is similarity negatively associated with frequency?
3. Within a fixed frequency stratum, is similarity positively associated with
   length? Marginal correlations alone cannot answer either question.
4. Does a structural pair outperform a vector-blind, frequency/length-matched
   unrelated pair? Is identical training incidence or initialization a better
   explanation than spelling by itself?

These are hypotheses, not a guarantee about every low-frequency/long n-gram.
No observation is removed because it contradicts a desired direction.

## Sources and coordinate spaces

- Nine original TinyShakespeare SGNS checkpoints: max orders 4/8/12, seeds
  3/7/11, independent learned rows, legacy input-gradient convention.
- Eighteen enwik8 SGNS checkpoints: the same orders/seeds, both legacy and
  standard average-derivative conventions. Use saved initialization and
  reproduce sampled center counts, verifying their original hashes.
- Six trained exact-key Qwen3-0.6B/Qwen3.5-4B checkpoints at 20,877 edits,
  seeds 3/7/11. Use canonical training-only vocabulary counts and original
  trained FP32 rows, excluding zero sentinels and budget padding. Recreate
  the explicitly seeded per-head initialization from the original source.

Character lengths and canonical token orders are separate domains. Qwen
comparisons stay within the same order and vector head: different heads have
independently learned projections, so cross-head cosine is not a valid parent
comparison. These small finite SGNS and frozen-backbone editing studies cannot
establish full-model pretraining or arbitrary-corpus generality.

## Pair definitions and sampling

Analyze three separate character relations: immediate parent/child (remove one
character from either end; deduplicate pairs), same-length prefix siblings and
same-length suffix siblings (share length-1 characters; different final/initial
character). Qwen uses the two sibling relations within each head.

Enumerate parent/child pairs. Sibling groups can be quadratic; sample uniformly
from the complete pair population using metadata-only indices, with at most
50,000 pairs per relation/head. Record population and sample counts. Sampling
uses a fixed seed, never vector values. Do not report sampled coverage as full
population coverage or count repeated model seeds as independent n-gram keys.

For each selected structural pair keep the first row as anchor, and replace
the second with a different row of exactly the same length and log2 frequency
bin. Try 64 vector-blind candidates and select the closest occurrence count,
with seeded tie breaking. Reject equal IDs and any immediate parent/child or
length-1 sibling relation to the anchor. Disclose unmatched pairs and actual
frequency balance. Structural/matched-control comparisons use the same matched
pair subset; overall structural distributions retain unmatched pairs.

## Measures and strata

Cosine and symmetric norm ratio min(norm_a,norm_b)/max(norm_a,norm_b) use float64
arithmetic on original FP32 bytes. Zero-norm pairs are undefined and separately
counted; they never pass a similarity gate. Primary gates are cosine >=0.99
and cosine >=0.99 plus norm ratio >=0.99. Also report quantiles, row L2
replacement error, initialization cosine and accumulated-update cosine when
initialization is available. Similarity does not mean original-byte equality.

Frequency is source training occurrence count including multiplicity. For
parent/child the primary frequency is the child count; also report parent/max
frequency and child/parent count ratio. For siblings use max(pair counts), so
"rare" requires both rows to be rare; retain min frequency as a diagnostic.
Use exact character length/token order and powers-of-two frequency bins.
Report source occurrence counts and actual sampled SGD exposures separately.

Within-length and within-frequency-bin Spearman coefficients are descriptive,
with groups of at least 50 valid observations. Publish every group and the
observation-weighted coefficient summary, for both continuous cosine and the
>=0.99 indicator. Do not turn these into independent-pair p-values or a causal
frequency effect. Report each seed, optimizer convention and head separately.

## Resource interpretation and claims

For character rows only, run the existing direct-to-retained prefix/suffix
representative policy at cosine/norm gates (0.99,0.99) and (0.99,0.95). Record
retained rows, actual packed table/map bytes, reconstruction error and occurrence
mass affected. This is approximate row storage, not original-row losslessness,
full-model quality evidence or a calibrated task-loss guarantee. The general
unrelated-vector analysis is not a new globally optimal clustering algorithm.

Keep supported existence, conditional association, mechanism and compression
claims separate. Mixed or opposite signs require narrowing/rejecting the
corresponding frequency/length hypothesis. No "first discovery" claim follows
from this experiment. All new work is CPU-only; original assets are read-only,
and remote writes stay in a new directory under the user's own workspace.
