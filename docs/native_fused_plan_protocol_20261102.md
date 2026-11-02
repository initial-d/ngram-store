# Production native-plan follow-up

Saved before native measurements. Reference is `tools/cotraining_groups.py`
at main `aa38f3371a1a8ac8e35dd258cb955688f6329fa0`. The new plan shares
original-ID counts with its certificate and avoids a whole-map representative
scan: certified groups contain every member, so their first observed ID is the
original representative. Immutable byte-backed partition arrays permit caching
the mapping digest once; native compilation reads the cached digest instead of
hashing/scanning the entire map on every query. Report this combined mechanism,
not only the integer-counting optimization.

Use the existing actual 1,855,907-row, 256-dimensional trained enwik8 subset
and its published 512 held-out registered-word trace. Reconstruct the trace
using the prior lexical/seed-42 selection and require its original flat-ID plus
offsets SHA256 to equal the previous report. Load the existing certified raw
native block store and verify its partition and manifest/table hashes. No
packing, changed vectors, new training or query selection is allowed.

Compare all plan arrays, summaries and native outputs byte-for-byte before
timing. Old/new use their own source-pinned partition implementations and the
same native C reducer, float64 weighted accumulation, final float32 output and
mean denominators. Measure old compilation, new compilation, precompiled replay,
old compilation plus query and new compilation plus query. Use one fresh CPU
process, a 4 MiB decoded cache, two warmups, 15 repetitions and seed-42 paired
interleaving. Record source/library/table/partition/trace hashes, raw samples,
p50/p95, OS/CPU/runtime and peak process RSS.

This is shared warm-CPU latency for the registered aggregate domain. It adds no
original-row-byte or quality guarantee, no OOV/fallback result, no comparison to
raw mmap's total query path, no isolated RSS reduction and no GPU claim. Preserve
all failures. The earlier FAISS preliminary retiming at `5dd15b4` regressed
direct latency; keep it as diagnostic evidence rather than a final speed claim.
