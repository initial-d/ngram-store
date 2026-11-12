# Construct certified incidence groups with disk-backed buffers

From a checkout containing this change, install the NumPy-only optional
interface (Python 3.10+):

```bash
python -m pip install '.[incidence]'
```

The distribution name remains `meft-memory`; both `ngram_store` and historical
`meft_memory` imports work. Torch is not required for this construction path.

Start with a frozen, row-contiguous little-endian FP32 table. Record every
registered aggregate query as a CSR bag: `ids.u64` contains original row IDs,
including duplicates; `offsets.u64` starts at zero and ends at the ID count.
Keep empty bags and the source ordering. Independently accessed word/padding
rows can be listed in an optional `protected.u64` file.

```bash
ngram-store compact-bags \
  --native \
  --input vectors.f32 --dim 256 \
  --bag-ids ids.u64 --bag-offsets offsets.u64 \
  --output compact --memory-mib 64 \
  --protected-ids protected.u64
```

Omit `--protected-ids` if no rows need separate protection. Construction
externally sorts complete incidence signatures, compares the full signature
after a hash match, and keeps unseen/protected rows separate. The disk-backed
FP64 accumulation follows ascending original row order before rounding to FP32.

With `--native`, the serving directory contains only `table.men`, `domain.npz`
and `manifest.json`. It streams dense group IDs into the original-ID native
map and writes raw blocks without a full descriptor/representative dictionary
or expanded canonical staging table. Intermediate maps/means are removed.
Open it with `CertifiedBagRows`, or run:

```bash
ngram-store query-bags --store compact --bags queries.json --mode mean
```

`queries.json` is a list of original-ID bags. Every query must satisfy the
certificate. [Release installation, offline example and PyTorch CPU integration](try_certified_memory.md).

Without `--native`, the historical construction output still contains:

- `partition/mapping.u64`, `sizes.u64`, `external.json`: disk-backed partition
  with array hashes;
- `means.f32`: one FP32 vector per **group ID**, ordered by the first original
  row in each group;
- `domain.npz`: the existing `BagPartition.load` format;
- `compact.json`: dimensions, domain and conversion contract.

The managed budget controls explicit working buffers and caches. It is not a
hard RSS, cgroup, disk or OS page-cache cap. Temporary files scale with the
input; place the output on a filesystem with enough capacity. Corpus/JSON/key
preparation is a separate stage and may still require substantial RAM.
[Measured RAM, time and disk costs](../../evidence/external_incidence_evidence_20261102/README.md).

For a streaming input iterator, construct and project separately:

```python
from ngram_store import fit_external

partition = fit_external(bags, nrows, "partition", memory_bytes=64 << 20)
partition.project_file("vectors.f32", 256, "means.f32", memory_bytes=64 << 20)
partition.save_npz("domain.npz")
```

Consume each bag iterator completely before advancing to the next. The caller's
own resident input arrays or registries are outside the working-buffer contract.

Runtime use retains the existing certificate and memory boundary:

```python
from ngram_store import BagPartition

runtime_partition = BagPartition.load("compact/domain.npz")  # resident O(N) map
if not runtime_partition.certifies(original_ids):
    raise ValueError("query is outside the certified aggregate domain")
```

Do not treat `means.f32` as an original-ID row table: its IDs are group IDs.
`CertifiedBagPlan` uses representative **original IDs** for an existing expanded
native certified artifact. Loading runtime maps and compiling query plans retain
their separate memory costs. `compact-bags --native` now streams the canonical
group table directly; the original `export_certified_bags` API still uses
resident maps/expanded staging.

This projection changes individual source vectors. Equal per-member counts
preserve sums in real arithmetic; arbitrary row lookup, incomplete groups,
independent gates and OOV queries do not receive that guarantee. The new builder
is byte-identical to the previous builder's partition/means, which is distinct
from source-row byte-preserving compression. Ordinary `ngram-store pack/query`
keeps its existing original-byte contract. See the
[co-training API and assumptions](cotraining_groups.md).
