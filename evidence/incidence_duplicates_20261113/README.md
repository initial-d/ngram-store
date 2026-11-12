# Duplicate rows are exactly the rows with identical incidence

With sum/mean pooling and plain sparse SGD, two rows that occur in the same
training bags with the same multiplicities receive identical gradients at every
step, so their difference never changes. With zero initialization they stay
byte-identical. This checks the converse on a real table: are *all* duplicates
explained by incidence?

AG News zero-init table from the [host query profile](../host_query_profile_20261112/README.md)
(6,989,611 trained word 1–4-gram rows, 10 epochs):

| | Groups |
| --- | ---: |
| Rows grouped by identical (document, multiplicity) sets | 883,125 |
| Rows grouped by identical bytes | 883,125 |
| Joint groups | 883,125 |

The two partitions are identical: rows are byte duplicates if and only if their
incidence matches. 78.6% of keys occur exactly once in training, which is why
most rows share a signature with others from the same document.

[`check.py`](check.py) · [`result.json`](result.json). Incidence signatures use an
order-free 64-bit hash of each row's (document, multiplicity) set; the equal group
counts rule out collisions merging distinct signatures.
