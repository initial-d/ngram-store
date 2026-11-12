# Prefix/suffix sharing stored without compression

Joint `.99/.99` DATrie sharing merges a row into its prefix/suffix
representative, which makes the two rows byte-identical. Those duplicates can be
stored with the deduplicated, uncompressed layout from the
[host query profile](../host_query_profile_20261112/README.md), so lookups copy
rows instead of decoding blocks. This checks what that keeps of the saving.

## Result

Nine trained SGNS checkpoints from the [tree-node mechanism study](../tree_node_quality_evidence_20261109/README.md)
(maximum order 12, average derivatives, initialization scale 1, seeds 3/7/11,
2k/8k/32k training steps; 18,137 × 128 FP32 n-gram rows each):

| Training steps | Distinct rows after sharing | Joint + dedup, uncompressed | Joint, XOR + zlib blocks (original) | Identity, XOR + zlib |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 97% | 1.4–1.5% saving | 7.2% | 5.9–6.0% |
| 8,000 | 86% | 12.5–12.8% | 17.9–18.2% | 6.6% |
| 32,000 | 56% | 42.3–42.5% | 45.9–46.1% | 7.8% |

Lookup p50 for 5,000 random 40-ID bags, relative to an in-memory array:

| Layout | vs raw |
| --- | ---: |
| Joint + dedup, uncompressed | 1.9–2.2× |
| Joint, XOR + zlib (original) | 122–127× |
| Identity, XOR + zlib (original) | 108–128× |

At 32k steps the uncompressed layout keeps 42.4% of the 46.0% saving and
removes the decoding cost. Every decoded row matched the original joint table
byte for byte. The relative change in held-out loss from joint sharing, taken
from the study's archived 2,000-resample intervals, stays within ±6×10⁻⁵ on the
test and confirmation segments, for all and OOV-only centers.

## Limits

The table is 9 MB and fits in CPU cache, so absolute lookup times say little
about large tables; the ratios show the decoding cost. Query bags are random
IDs, not the study's text traces. The savings are for the n-gram table only.
The longer-trained points also have higher source loss (see the study), so this
shows what training does to structure, not a recommended training length.

## Setup

[`structure_remote.py`](structure_remote.py) on the grid machine (AMD EPYC 7713,
one pinned core, Python 3.12, NumPy 1.26) with the v0.5.0 runtime: decode the
study's identity and joint tables, re-pack each with `pack_native(...,
deduplicate=True, compress=False)`, time lookups. Raw output:
[`results.json`](results.json).
