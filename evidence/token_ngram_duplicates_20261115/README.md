# Duplicate rows in a token-level memory

The deduplication results elsewhere in this archive come from word n-grams over
documents. A token-level memory for a language model is a different object: the
keys are tokenizer ids, the bags are fixed-length windows of a long stream, and
the table is much larger. This checks whether the same redundancy appears there.

Answer: the duplicate fraction is high, but almost all of it comes from keys
that occur in a single training window, and it requires zero initialization. That
part of the table is what a count threshold would discard anyway, so the
deduplication saving on a token stream does not survive a comparison with
pruning. The result is recorded here as a boundary on
[the host query profile](../host_query_profile_20261112/README.md) rather than as
a new application.

## Setup

WikiText-103 raw training split, tokenized with the Qwen3-0.6B tokenizer
(151,669 entries), split into non-overlapping 512-token windows. Keys are token
n-grams of orders 1-4 with ids offset by one, so zero marks padding. For each key
the incidence set holds the (window, multiplicity) pairs it occurs in, hashed
order-free; two keys with the same signature receive identical gradients at every
step, so with zero initialization their rows stay byte-identical
([Proposition 1](../incidence_duplicates_20261113/README.md)).

No training is involved. [`count.py`](count.py) recomputes everything from the
token stream.

## Result

Full corpus, 124,245,019 tokens, 242,666 windows:

| Order | Distinct keys | Rows stored | Duplicate fraction | Keys in one window | Keys in several |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 62,775 | 61,094 | 2.68% | 6,100 | 56,675 |
| 2 | 8,859,857 | 3,997,891 | 54.88% | 4,969,709 | 3,890,148 |
| 3 | 38,246,489 | 8,810,239 | 76.96% | 28,352,437 | 9,894,052 |
| 4 | 69,411,465 | 8,088,490 | 88.35% | 58,808,660 | 10,602,805 |

At order 4, 84.7% of keys appear in exactly one window. Those keys collapse to a
few hundred rows per window, because within a window every such key sees the same
gradient. The remaining 10.6M keys, which appear in several windows, deduplicate
by only 27.2%. The hash count and an exact distinct count agree on every prefix
checked.

## Thresholds and scale

Deduplicating only the keys that survive a count threshold, full corpus:

| Minimum windows | 2-gram | 3-gram | 4-gram |
| ---: | ---: | ---: | ---: |
| 2 | 4.99% | 14.61% | 27.15% |
| 3 | 1.30% | 4.70% | 10.46% |
| 5 | 0.59% | 2.24% | 5.43% |
| 10 | 0.30% | 1.23% | 3.20% |
| 100 | 0.02% | 0.14% | 0.58% |

The saving falls away as the threshold rises, because a key that appears in many
windows rarely shares its exact incidence pattern with another key. Scaling the
corpus from 16M to the full 124M tokens changes the picture only slightly: at
order 4 the multi-window saving moves from 24.6% to 27.2%, and the total
duplicate fraction falls from 92.0% to 88.4%.

Window length from 512 to 4096 tokens changes the order-4 total by less than two
points (92.0% to 93.5%). Repeated epochs leave the partition unchanged: every
multiplicity is multiplied by the same factor, which only renames groups.

## What this rules out

A token-level pooled n-gram memory cannot be justified by deduplication alone. Its
large headline fraction is the collapse of single-window keys, which pruning
removes for free, and the part that survives pruning is a quarter of the 4-gram
table and nothing at the orders that dominate a token vocabulary. The layout
results for word tables in the
[host query profile](../host_query_profile_20261112/README.md) still apply to
whichever rows are kept; what does not transfer is the size of the saving.

All of this holds only under zero initialization. With random initialization the
rows of two keys with equal incidence keep a constant difference rather than
being equal, so byte deduplication finds nothing.

## Reproducing

```
pip install numpy pyarrow tokenizers
python count.py --tokens 16000000 --window 512 --stride 512 --orders 1,2,3,4
```

The corpus is downloaded from `Salesforce/wikitext` (`wikitext-103-raw-v1`, the
two training parquet shards) and the tokenizer from `Qwen/Qwen3-0.6B`; paths are
at the top of the script. The full-corpus run takes about five minutes and peaks
at 7.2 GiB. [`full-corpus.json`](full-corpus.json) and
[`min-count-sweep.json`](min-count-sweep.json) hold the published numbers.
