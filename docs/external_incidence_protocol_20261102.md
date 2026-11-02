# External incidence construction protocol

Saved before remote construction measurements on 2026-11-02. CPU only; no GPU
inspection, stopping, training or new external data fetch.

Use the existing finite enwik8 SGNS table: 1,855,907 actually updated rows,
256 FP32 dimensions, source SHA256
0e5d4be0b156c5e9f0d745292876e48f78772c599d97902619123512955d82fc.
Reuse the cached registered word list and original n-gram key ordering. Prepare
raw uint64 CSR IDs/offsets once in a separate process; check its full domain
against the prior 142,308-bag certificate, including multiplicities:
domain SHA256 4faabf557ad3bbc67da96a5d71f0cc7a78b1066713efa51f7d934aa9bcf01a5d.
Record preparation time/RSS separately. It is not a bounded corpus/tokenizer
pipeline. Do not include private text or weights in public artifacts.

Implement fixed-buffer external sorts by bag/row and row/bag, preserving
duplicates and gaps. Index signature digests in a disk-backed SQLite cache;
compare complete signatures to resolve every candidate collision. Number groups
in ascending first-original-row order, exactly as BagPartition.fit. Empty and
protected rows remain individual groups. No resident signature/map/group array
may scale with total rows or occurrences. Run-path enumeration must use bounded
fan-in passes. Temporary disk may scale with input size.

Aggregate original rows in ascending ID order using float64 accumulators on
disk and a fixed-capacity cache, then divide and round to FP32. Match the old
BagPartition.means output bytes. Export the existing domain.npz reader format
without resident mapping copies. Runtime materialization, native packing and
corpus-to-CSR preparation retain their separate memory contracts.

Run each build in a fresh single-threaded CPU worker. Compare current in-memory
fit plus means with external construction at managed buffer budgets 64/128/256
MiB on the full table. Add actual 250,000 and 1,000,000 original-row prefixes,
filtering each registered bag to that prefix, at 64 MiB and with a fresh
in-memory baseline. These are real-table subsets, not independent training or
synthetically replicated rows. Keep every original registered bag including
empty ones. Publish every run, including failures and slower construction.

Measure process peak RSS, stage and complete wall time, complete output bytes,
and sampled workspace logical disk peaks at run/merge/stage boundaries. Managed
budget is not a hard process/cgroup RSS or OS page-cache cap; runtime and SQLite
overhead remain measured. Record temporary accumulator logical size separately.
Do not infer a hard RSS guarantee merely from a small configured cache or
disk-backed maps. No cold-page/exclusive CPU or monetary/energy claim.

Require exact domain/mapping/group-size and all FP32 group-mean bytes against
the fresh legacy baseline at every scale/budget. On the full table also compare
the earlier certificate and unchanged 512-bag native plan/output bytes in a
separate verifier. Verification/runtime RSS is excluded from construction RSS
and reported separately. Preserve the maximum observed bag length and all input,
code, SQLite, package, hardware, command, source-SHA and output hashes.

Operational correction after the first 250k attempt: per-bag temporary-file
truncate/seek/flush incurred substantial I/O waiting. Preserve the interrupted
attempt and its completed legacy result. Use a bounded byte payload of at most
budget/64 for small bags, spilling larger bags to a visible owned file. This
changes temporary I/O, not signatures, group numbering, averaging or inputs.
Rerun the complete grid in a new source snapshot with the same cached CSR and
all declared budgets. Include a forced-spill correctness test.
