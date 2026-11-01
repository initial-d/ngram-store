# Certified co-training groups

This optional **model-side** path detects exact equality of embedding-bag
incidence columns. It complements NGramStore's existing original-ID/row-byte
storage path; it does not turn arbitrary vector replacement into lossless
compression. [Measured evidence](cotraining_evidence_20261031/README.md).

## Why final cosine can miss the opportunity

Let `A[b,i]` be the multiplicity of original row `i` in registered bag `b`.
Rows in the same group have identical complete `(bag ID, multiplicity)` lists.
For a model consuming their sum/mean, the chain rule gives identical per-row
gradients. With the demonstrated sparse SGD rules and equal optimizer settings:

```
e_i(t) = e_i(0) + U_g(t)
e_i(t) - e_j(t) = e_i(0) - e_j(0)
```

Independent initialization can therefore retain a low final cosine even when
actual updates align. High cosine is neither necessary for certified bag
equivalence nor sufficient for original-vector byte equality.

For group mean `mu_g`, every certified query has equal multiplicity `c_g` for
each member. In real arithmetic:

```
sum_i A[b,i] * e_i = sum_g c_g * |g| * mu_g
```

The within-group zero-sum residual does not affect those aggregates. Replacing
rows by their group mean changes individual vectors and may change an
uncertified query. Floating-point reduction/rounding is measured separately.

The condition is also **necessary for a weight-independent guarantee**: if two
members have different query counts, put `+1` and `-1` at those members of a
one-dimensional source table and zero elsewhere. Their group mean is zero,
but the query sum is their nonzero count difference. Thus equal per-member
counts describe the largest universally safe bag domain for this projection in
real arithmetic. A rejected query can still happen to agree on a particular
trained table. This is an application of classical linear algebra, rather than
a claim to invent semantic-preserving network reduction.

`partition.explain_query(bag)` returns either a uniform certificate or a
violating group and sparse algebraic counterexample. This diagnostic uses no
source vectors and has no semantic-quality estimate. Exclude padding as the
consumer does. An incomplete group may require scanning the ID map in bounded
chunks; use diagnostics outside the hot query path. The unchanged `certifies`
method and compiled native schedule remain the serving path.

For multiple queries, `partition.certifies_many(bags, padding_idx=0)` returns
the same Boolean decisions using batched integer count comparisons.
`CertifiedBagPlan` uses this path automatically. On nine real cached checkpoints,
batch certification is about 7.3–7.8× faster and native-plan construction about
3.1–3.4× faster, with every resulting plan array byte-identical. These are
shared-CPU compilation results, not full-query or reducer speedups;
[paired source-pinned measurements](faiss_cotraining_evidence_20261101/README.md).

This shared-gradient identity follows from the chain rule. Novelty relative to
parameter tying, duplicate-feature elimination, embedding factorization and
existing storage methods still needs a literature comparison. The contribution
to evaluate is the certified parameterization/export/query method and its
measured scope, rather than claiming discovery of backpropagation itself.
The [contribution claims](cotraining_contribution_claims.md) separate supported
method/results from engineering features, prior foundations and unverified
research opportunities.

## API and scope

With `meft-memory[torch]`, import through either `ngram_store` or the historical
`meft_memory` namespace:

```python
from ngram_store import (
    BagPartition, grouped_embedding_bag, export_certified_bags, CertifiedBagRows,
)

partition = BagPartition.fit(training_bags, row_count,
                             protected=word_and_padding_ids)
model = grouped_embedding_bag(initial_rows, partition, padding_idx=0)
# Train this mutable PyTorch module, then freeze its rows for export.
export_certified_bags(model.expanded_rows().detach().cpu().numpy(), partition,
                      output, compress=False)
with CertifiedBagRows(output) as store:
    plan = store.compile(query_bags, padding_idx=0)
    values = store.evaluate(plan, mean=True)
```

The [executable example](../examples/cotraining_groups.py) is a correctness
fixture; it is not experiment data. Save the partition and model `state_dict`
for training checkpoints rather than pickling the factory's local module class.

- Signature equality includes **all registered bags and multiplicities**.
  Shared spelling, one common host word or potential membership alone is weaker.
  Empty/unobserved rows remain singletons; protect independently accessed roles.
- A group of `k` rows has its shared gradient divided by `k`, preserving the
  demonstrated per-original-row SGD update. Simply aliasing parameters changes
  the effective learning rate. No row-specific dropout, regularizer, global
  gradient clipping or independently gated/head-specific coefficients are
  covered by the current proof/tests.
- Queries require equal per-member multiplicity within every used group. This
  can certify some new combinations as well as original bags; incomplete groups
  are rejected. Across LLM heads/layers or separately transformed memory fields,
  fit separate role-specific domains and establish coefficient equality first.
- Compilation checks the domain once and constructs original representative
  IDs, multiplicities and offsets for the existing C weighted reducer. Include
  plan memory/build cost; its FP64 reduction is not bit-identical to FP32 sums.
- `compress=False` stores raw native blocks with the same reader/forwarding
  format. Structural sharing still removes rows. Default row packing retains
  its existing compression behavior; original byte-preserving packers remain
  available for unrestricted row queries.
- A grouped model can export its own already-shared rows with the ordinary
  byte-preserving packer. That preserves the **grouped checkpoint's** bytes;
  equivalence to an independently initialized model outside the certified
  domain remains a separate question.

For strictly known-word inference, directly precomputing the complete word
aggregates is smaller/faster on these references. Grouped parameters instead
retain the demonstrated training-update geometry and a broader certified query
space. Keep the stronger precomputation baseline in serving comparisons.

Construction currently uses Python incidence lists/dictionaries, O(N) maps and
FP64 group accumulators. The large study peaks at about 5.4 GiB RSS; this is not
a bounded-memory structural builder. Source initialization is released after
conversion, and reordered native ID maps use an additional `8*N` bytes to avoid
per-row filesystem seeks. C store handles remain frozen and single-worker.
