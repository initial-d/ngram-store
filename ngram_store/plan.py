"""Choose a native layout for a frozen float table from its duplicate rows.

Lookups on compressed blocks decode whole blocks, which is slow when the IDs of
one query are scattered (see evidence/host_query_profile_20261112). Layouts that
store rows uncompressed only copy bytes. So the planner first looks for byte
duplicates, which save space without any decoding, and only suggests block
compression when the caller asks to optimize for size.
"""
import math
from pathlib import Path

HEADER_BYTES = 56          # MEFTNAT1 header
DESCRIPTOR_BYTES = 32      # per block
MAP_BYTES_PER_ROW = 8      # forwarding map, present when deduplicating
UNCOMPRESSED_BLOCK_ROWS = 256
COMPRESSED_BLOCK_ROWS = 256


def _row_hashes(source, row_bytes, chunk_rows):
    """64-bit FNV-style hash of every row, computed in chunks."""
    import numpy as np
    rows = source.stat().st_size // row_bytes
    table = np.memmap(source, dtype=np.uint8, mode="r", shape=(rows, row_bytes))
    words = -(-row_bytes // 8)
    hashes = np.empty(rows, dtype=np.uint64)
    prime, basis = np.uint64(1099511628211), np.uint64(14695981039346656037)
    with np.errstate(over="ignore"):
        for begin in range(0, rows, chunk_rows):
            part = np.asarray(table[begin:begin + chunk_rows])
            if row_bytes % 8:
                padded = np.zeros((len(part), words * 8), dtype=np.uint8)
                padded[:, :row_bytes] = part
                part = padded
            columns = part.view("<u8")
            h = np.full(len(part), basis, dtype=np.uint64)
            for j in range(words):
                h ^= columns[:, j]
                h *= prime
            h ^= h >> np.uint64(29)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(32)
            hashes[begin:begin + len(part)] = h
    return table, hashes


def _uncompressed_bytes(rows, unique, row_bytes, deduplicate):
    blocks = math.ceil(unique / UNCOMPRESSED_BLOCK_ROWS)
    return (HEADER_BYTES + (MAP_BYTES_PER_ROW * rows if deduplicate else 0)
            + DESCRIPTOR_BYTES * blocks + unique * row_bytes)


def _compression_ratio(table, hashes, row_bytes, codec, sample_blocks, distinct, seed):
    """Estimated compressed/raw ratio of 256-row blocks, from a sample."""
    import numpy as np
    from meft_memory._runtime.block_compression import compressor
    encode = compressor(codec)
    rng = np.random.default_rng(seed)
    rows = len(hashes)
    if distinct:   # blocks of distinct rows, as stored after deduplication
        picked = rng.choice(rows, size=min(rows, sample_blocks * COMPRESSED_BLOCK_ROWS * 2), replace=False)
        _, first = np.unique(hashes[picked], return_index=True)
        picked = np.sort(picked[first])[:sample_blocks * COMPRESSED_BLOCK_ROWS]
        groups = [picked[i:i + COMPRESSED_BLOCK_ROWS] for i in range(0, len(picked), COMPRESSED_BLOCK_ROWS)]
    else:          # contiguous blocks, as stored without deduplication
        starts = rng.choice(max(1, rows - COMPRESSED_BLOCK_ROWS + 1), size=min(sample_blocks, rows), replace=False)
        groups = [np.arange(s, min(s + COMPRESSED_BLOCK_ROWS, rows)) for s in starts]
    raw = stored = 0
    for group in groups:
        block = np.ascontiguousarray(table[group]).tobytes()
        raw += len(block)
        stored += min(len(block), len(encode(block)))
    return stored / raw if raw else 1.0


def plan_layout(source, dim, *, dtype_bytes=4, prefer="speed", codec="zstd",
                sample_blocks=64, chunk_rows=1 << 18, seed=0):
    """Return size estimates for candidate layouts and a recommendation."""
    if prefer not in ("speed", "size"):
        raise ValueError("prefer must be speed or size")
    source = Path(source)
    row_bytes = dim * dtype_bytes
    rows, tail = divmod(source.stat().st_size, row_bytes)
    if type(dim) is not int or dim <= 0 or not rows or tail:
        raise ValueError("input must contain nonempty complete rows of dim values")
    import numpy as np
    table, hashes = _row_hashes(source, row_bytes, chunk_rows)
    unique = int(np.unique(hashes).size)
    raw_bytes = rows * row_bytes
    block_bytes = UNCOMPRESSED_BLOCK_ROWS * row_bytes
    base = f"ngram-store pack --input {source} --dim {dim} --output OUTPUT --no-compress --max-block-bytes {block_bytes}"
    candidates = [
        {"layout": "raw", "lookup": "copy", "bytes": _uncompressed_bytes(rows, rows, row_bytes, False),
         "command": base},
        {"layout": "dedup", "lookup": "copy", "bytes": _uncompressed_bytes(rows, unique, row_bytes, True),
         "command": base + " --deduplicate"},
    ]
    if prefer == "size":
        try:
            generic = _compression_ratio(table, hashes, row_bytes, codec, sample_blocks, False, seed)
            distinct = _compression_ratio(table, hashes, row_bytes, codec, sample_blocks, True, seed)
        except ValueError:
            codec = "zlib"
            generic = _compression_ratio(table, hashes, row_bytes, codec, sample_blocks, False, seed)
            distinct = _compression_ratio(table, hashes, row_bytes, codec, sample_blocks, True, seed)
        blocks = lambda n: DESCRIPTOR_BYTES * math.ceil(n / COMPRESSED_BLOCK_ROWS)
        candidates += [
            {"layout": codec, "lookup": "decode", "estimated": True,
             "bytes": int(HEADER_BYTES + blocks(rows) + rows * row_bytes * generic),
             "note": "Python API only: pack_native(..., deduplicate=False, codec='%s')" % codec},
            {"layout": "dedup+" + codec, "lookup": "decode", "estimated": True,
             "bytes": int(HEADER_BYTES + MAP_BYTES_PER_ROW * rows + blocks(unique) + unique * row_bytes * distinct),
             "note": "Python API only: pack_native(..., deduplicate=True, codec='%s')" % codec},
        ]
    for c in candidates:
        c["saving_vs_source"] = round(1 - c["bytes"] / raw_bytes, 6)
    pool = candidates if prefer == "size" else [c for c in candidates if c["lookup"] == "copy"]
    best = min(pool, key=lambda c: c["bytes"])
    if best["layout"] == "raw":
        reason = ("Few duplicate rows: deduplication would not pay for its row map. "
                  "Keep the table uncompressed (or serve the float file with mmap).")
    elif best["layout"] == "dedup":
        reason = ("%.1f%% of rows are duplicates. Deduplication saves space and lookups "
                  "still copy rows without decoding." % (100 * (1 - unique / rows)))
    else:
        reason = ("Smallest estimated size. Lookups decode whole blocks and can be much "
                  "slower than raw on scattered IDs; measure on your trace.")
    return {"rows": rows, "row_bytes": row_bytes, "source_bytes": raw_bytes, "unique_rows": unique,
            "duplicate_fraction": round(1 - unique / rows, 6), "prefer": prefer,
            "candidates": candidates, "recommended": best["layout"], "reason": reason,
            "note": "Unique rows are counted with a 64-bit row hash; pack verifies every row."}
