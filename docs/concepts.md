# Concepts

The terms below come up throughout the guides and the evidence reports.

## Tables, IDs and bytes

**Frozen table.** A trained embedding matrix that no longer changes: `rows × dim`
values, usually float32. NGramStore only serves frozen tables; training stays in
the model's own code.

**Original row ID.** The integer the model already uses to address a row, whether
it comes from a vocabulary, a DATrie or a hash function. NGramStore keeps these
IDs unchanged, including their order and duplicates within a query.

**Row bytes.** The exact bit pattern of a stored row. Lossless modes return the
same bytes that went in, including signed zeros and NaN payloads; `query --hex`
shows them.

## Lossless and approximate

| Mode | Changes vectors? | What it does |
| --- | --- | --- |
| `pack` (default) | No | Compresses rows in blocks |
| `pack --deduplicate` | No | Stores identical rows once and adds a row map |
| Structural `layout` / `exact` | No | Reorders rows by key structure; `exact` also shares byte-identical prefix/suffix rows |
| Structural `joint` | Yes | Merges prefix/suffix neighbours whose cosine and norm ratio both pass a threshold (default .99) |
| FP16 / INT8 | Yes | Quantizes rows, then stores the encoded bytes losslessly |
| Certified bag groups | Yes, per row | Replaces rows by group means; sums over complete groups stay exact |

Lossless means the stored bytes come back unchanged. It does not by itself say
anything about speed or model quality; those are measured separately.

## Native store

**Native store.** A directory with `table.men` (the `MEFTNAT1` binary file),
`store.json` (shape, hashes, mode) and `embedding.json` (PyTorch adapter metadata).

**Block.** Rows are grouped into blocks (256 rows by default, fewer if
`--max-block-bytes` caps the decoded size). Each block is stored raw, zlib, or
XOR-with-previous-row + zlib, whichever is smallest. A query decodes whole blocks,
so smaller blocks make sparse lookups cheaper and compression worse.

**Forwarding map.** When rows are deduplicated or reordered, an 8-byte-per-row
map sends each original ID to its stored row. Without either, format version 3
omits the map and saves `8 × rows` bytes.

**Decoded-block cache.** An optional byte budget for keeping decoded blocks (or
only the queried rows) between calls. The budget limits cached payload, not the
process's total memory (RSS).

## Keys and structure

**Registered keys.** A one-to-one map from row IDs to strings, such as character
n-grams. Only registered keys enable the structural layer; rows addressed by hash
have no meaningful string relations, because one bucket can hold unrelated n-grams.

**DATrie.** A double-array trie over the registered keys. It gives exact key → ID
lookup and lets the planner walk prefix and suffix chains.

**Prefix/suffix representative.** When an n-gram and a shorter n-gram on its
prefix or suffix chain have the same bytes (`exact`) or pass the cosine and norm-ratio test
(`joint`), the child can point to the parent's row.

**Mark-compact export.** After representatives are chosen, the rows still in use
are copied into a dense table and the forwarding map is rewritten, the same steps
a mark-compact garbage collector uses to defragment a heap.

## Hash buckets vs distinct rows

**Hash buckets.** A fixed number of rows; each n-gram is hashed to one. Memory is
bounded, but unrelated n-grams that collide share a row and its updates.

**Distinct rows (exact identity).** One row per registered n-gram. No collisions,
but the table grows with the vocabulary. Whether that pays off is task-dependent;
see [evidence section 1](../evidence/README.md#1-does-exact-n-gram-identity-beat-hash-buckets).

**Necessity gate.** A check used in the LLM experiments: does the trained memory
beat cheaper controls (memory off, one learned constant per head, a small adapter)
on held-out data? If not, distinct rows are not needed for that task, however well
they are stored.

## Bags and certified groups

**Bag.** A list of original IDs that a model reduces by sum or mean, e.g. the
n-grams of one word in FastText. A bag may repeat IDs.

**Incidence signature.** For each row, the list of `(bag, multiplicity)` pairs it
appears in across a fixed set of registered bags.

**Certified group.** Rows with identical incidence signatures. Under sum/mean
training with plain sparse SGD they receive identical updates, so storing their
mean keeps every complete-group bag sum exact in real arithmetic, while individual
rows change. A bag is certified when, for each group it touches, every member
appears the same number of times. Other bags are rejected.

**Protected IDs.** Rows that are also read on their own (whole words, padding);
they stay in groups of one.

**Disk-backed (external) incidence build.** Builds signatures and group means
through files and a fixed buffer budget, instead of holding the whole table and
all signatures in memory.

**Budgeted partial query.** An opt-in path that also accepts uncertified bags when
a precomputed bound on the L2 error against the source vectors fits a given budget,
e.g. `--max-relative-l2 0.01`.

## Reading the evidence

**Protocol.** Written and committed before a run; fixes data, seeds, baselines and
pass/fail rules. **Verifier.** `verify.py` recomputes the published tables from
the archived results. **Controls.** Baselines run on the same data and trace:
raw mmap, dense in-memory lookup, FAISS codecs, hash tables of matched size.
