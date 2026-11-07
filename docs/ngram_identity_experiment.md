# Collision-free canonical n-gram rows versus shared hash buckets

The previous MultiCounterFact small/mid/large controls all used hash buckets.
They did not test a vocabulary with an independent row for every n-gram.
This [separate protocol](../benchmarks/ngram_identity_v1.json) directly changes
parameter tying while retaining the original Engram tokenizer normalization,
padding, orders 2/3/4, two 32-dimensional heads, projections, gates and convolution.

Build the key vocabulary only from edited rewrite training inputs before the
terminal EOS output. Never insert paraphrase, neighborhood or evaluation-answer
keys. Exact keys are chronological canonical token tuples. Engram's compressed
tokenizer still merges some raw token identities: this is a bucket-collision
control after canonical normalization, not a raw-token identity experiment.

For each order with U training keys, the hash control uses Engram's consecutive
prime bucket rule starting at U. The exact control assigns one distinct row
per key/head, then pads with never-addressed rows to match hash parameter and
physical row-byte budgets exactly. It starts each key vector as a clone of its
corresponding hash bucket vector. All nonrow weights and effective initial
embeddings match by seed. Subsequent training lets exact keys diverge, whereas
hash updates remain pooled. Extra padding is not a real vocabulary entry.

The primary comparison uses zero embeddings for unseen canonical keys in both
variants. Frozen zero rows prevent learning an extra unknown-key representation.
Separately evaluate the same trained hash checkpoint with ordinary open-vocabulary
bucket lookup, which may generalize better through sharing. Report both comparisons;
a win conditional on matched vocabulary coverage is not automatically a win
against open-vocabulary hashing. Count individual-head and joint-head address
collisions separately: two heads can distinguish keys even when one head shares.

Two models, two edit counts and three seeds yield 24 trained models. The entire
original Qwen backbone stays frozen to isolate memory identity; only the branch
trains for twenty fixed epochs. This schedule is based on the previously disclosed
frozen-backbone probe, not selected on these new test results. Reuse byte-checked
frozen prefixes with the same batch shapes and mask duplicate fillers. Retain
every official MultiCounterFact case and ES/PS/NS, harmonic score and full-target
teacher-forced accuracy. Counterfactual editing is not verified factual knowledge
or language-model pretraining. Generation metrics remain outside scope.

Before training, verify real-model paired initial logits, sixteen actual question
prefixes per model, cached/direct masked gradients, nonrow gradient agreement,
hash-gradient pooling over exact keys, and frozen-backbone hashes after an Adam
step. The gradient pooling check uses disclosed floating-point tolerance;
initial effective logits are checked byte for byte. These checks do not establish
a trained quality advantage.

```bash
python -m tools.counterfact_identity build --help
python -m tools.counterfact_identity check --help
python -m tools.counterfact_capacity train --help
```

Keep the vocabulary, delta and model recipe together for reproducibility.
Reports count the padded row table and complete delta/vocabulary/recipe bytes;
the shared immutable base is still required. Canonical-key files and indexes
have costs, while ordinary open-vocabulary hash lookup does not need the full
training-key dictionary. The [complete 24-model results](ngram_identity_evidence_20261107/README.md)
support an independent-row quality advantage at 20,877 edits under this protocol;
some 1,024-edit intervals cross zero. Native compression/query superiority and
full-model pretraining remain untested by this experiment.

The finite task-specific queue preserves the previous queue's active GPUs and
checks compute-process occupancy before launching. Only a separately audited,
authorized occupancy helper can terminate known occupancy programs. Unknown
processes and all training/evaluation processes are retained. Files and caches
stay under `/llm-align/duyimin/`.
