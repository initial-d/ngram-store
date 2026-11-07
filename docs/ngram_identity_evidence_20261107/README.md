# Full canonical n-gram rows vs shared hash buckets

All 24 trained runs and both preflights completed. Both queues and occupancy helpers exited. This final audit used CPU only.

Official MultiCounterFact counterfactual editing; frozen Qwen backbones; seeds 3/7/11; 20 fixed epochs. Exact and hash match row bytes, parameters, initial effective vectors and training exposures. The same hash checkpoint also receives native open-vocabulary evaluation. Complete vocabulary/delta/recipe costs are in the reports.

H is the harmonic mean of seed/case-pooled ES, PS and NS. Scores are percentages; deltas and 95% pointwise intervals are percentage points from 2,000 paired seed/case bootstrap replicates.

| Model | Edits | Hash control | Exact H | Hash H | H delta [95% interval] | NS delta |
|---|---:|---|---:|---:|---|---:|
| qwen06 | 1,024 | hash | 65.06 | 63.94 | +1.12 [+0.20, +2.10] | +1.55 |
| qwen06 | 1,024 | hash-native-oov | 65.06 | 64.68 | +0.38 [-0.59, +1.31] | +1.06 |
| qwen06 | 20,877 | hash | 71.72 | 71.31 | +0.41 [+0.04, +0.86] | +0.59 |
| qwen06 | 20,877 | hash-native-oov | 71.72 | 69.43 | +2.29 [+1.90, +2.73] | +3.02 |
| qwen35 | 1,024 | hash | 64.17 | 63.83 | +0.33 [-0.56, +1.17] | -0.32 |
| qwen35 | 1,024 | hash-native-oov | 64.17 | 62.40 | +1.76 [+0.94, +2.62] | -0.80 |
| qwen35 | 20,877 | hash | 73.41 | 72.17 | +1.24 [+0.87, +1.68] | +1.45 |
| qwen35 | 20,877 | hash-native-oov | 73.41 | 69.36 | +4.05 [+3.65, +4.40] | +1.72 |

At 20,877 edits both models pass the preregistered quality gate against both hash controls: positive H interval and no NS regression greater than 2 percentage points, including the conservative NS interval check. At 1,024 edits, Qwen3-0.6B versus native hashing and Qwen3.5-4B versus matched-coverage hashing have intervals crossing zero.

These results support an independent-row quality advantage in this frozen-backbone editing experiment. They do not establish full-model pretraining superiority, native serving/compression dominance or naturally occurring 99% cosine similarity. Three seeds and related subjects limit inference. Intervals are pointwise and uncorrected across eight comparisons.

The audit verified source and input hashes, final-epoch completeness, case/split identity, NLL-based success labels, all reported metrics, paired row budgets, optimizer steps, training exposures and frozen-weight hashes. This archive includes 24 reports, four vocabulary reports, two preflights, analysis code/logs, input hash manifests and paired test arrays. Original loss-level scalars and model deltas remain on the remote machine under the user task directory.

[Full analysis](final-analysis.json) · [CPU analysis script](analyze.py) · [Paired test arrays](paired-test-arrays.npz)

## Verify and reproduce

```bash
python -S docs/ngram_identity_evidence_20261107/verify.py
python docs/ngram_identity_evidence_20261107/verify.py --bootstrap
```

The first command uses only the standard library. The second needs NumPy and checks paired test-case arrays against reported means, then reproduces all eight bootstrap intervals from the archived pure statistics functions. Neither command needs GPUs, model weights or Torch. The original remote analysis script is retained unchanged as provenance, including its task-specific paths.

Training/preflight source: `425c1876614b9533ecf3b5e092e1dbed0bfd7651`; vocabulary-build source: `f4217e2127a4b24631f6dbf1bb30ab2578018a4a`. All reported runtime source files are archived. Environment, exact training commands, hardware, optimizer exposures, row bytes, complete bundle bytes and timings are retained in the 24 run reports. Hashes of the original remote loss-level files and model deltas are recorded, but those files are not bundled here.
