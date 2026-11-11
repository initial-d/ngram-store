"""Native C ABI bindings and exporter for the versioned MEFTNAT1 format."""
import ctypes
from contextlib import contextmanager
import os
from pathlib import Path
import struct
import stat
import tempfile
import zlib

HEADER = struct.Struct("<8s6Q")
DESC = struct.Struct("<4Q")
ROW = struct.Struct("<Q")


def resolve_block_rows(row_bytes, block_rows=256, max_block_bytes=None):
    """Cap decoded block width for sparse access; an indivisible row can exceed it."""
    if type(row_bytes) is not int or row_bytes <= 0 or type(block_rows) is not int or block_rows <= 0:
        raise ValueError("row bytes and block rows must be positive integers")
    if max_block_bytes is None:
        return block_rows
    if type(max_block_bytes) is not int or max_block_bytes <= 0:
        raise ValueError("max block bytes must be a positive integer")
    return min(block_rows, max(1, max_block_bytes // row_bytes))


@contextmanager
def _atomic_output(output):
    """Publish a complete same-filesystem file; existing mmap handles keep theirs."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent,
                                         prefix="." + output.name + ".", suffix=".tmp",
                                         delete=False) as dest:
            temporary = Path(dest.name)
            yield dest
            dest.flush()
            os.fsync(dest.fileno())
            # Preserve regular-file access permissions when replacing a table.
            # New artifacts keep the owner's tempfile permissions.
            if output.is_file() and not output.is_symlink():
                os.chmod(temporary, stat.S_IMODE(output.stat().st_mode))
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def pack_native(source, output, row_bytes, block_rows=256, deduplicate=True,
                predictor="none", row_order=None, *, representatives=None, max_block_bytes=None,
                compress=True, implicit_identity=True, codec='zlib'):
    """Stream blocks into a spool; exact dedup index still grows with unique rows.

    deduplicate=False avoids the unbounded unique-row dictionary for large raw
    tables. With no row_order/representatives it writes version 3, omitting the
    redundant identity map (8 bytes per row). Original IDs and bytes stay exact.
    implicit_identity=False exports versions 1/2 for older readers. Deduplicated,
    reordered and representative exports retain explicit forwarding in v1/v2.
    Incompressible blocks are stored raw, avoiding codec expansion per block.
    predictor="xor" tries previous-row XOR + zlib as a third block candidate
    and writes format version 2 when an explicit map is needed. Optional
    row_order is a full permutation of original IDs; its validation uses O(n)
    memory and reading becomes nonsequential. Reordered identity exports stage an
    additional 8 * nrows bytes to write the ID map sequentially, avoiding one
    small seek/write per row on remote filesystems. Original query IDs stay unchanged.
    Optional representatives is an idempotent original-ID -> original-ID map.
    Only marked representative rows are read and packed, then the original-ID
    forwarding map is rewritten to dense physical IDs. This performs actual
    mark/compact export without staging an expanded reconstructed table. The
    caller owns its similarity/byte-preservation contract: nonidentical aliases
    change original weights, even though decoding preserves representative bytes.
    The complete table is flushed/fsynced into a sibling temporary file before
    atomic replacement. Failed construction/publication preserves the old file;
    readers already mmap-ing it continue reading the old inode. This covers this
    single file, not multi-file adapter metadata or concurrent-writer coordination.
    compress=False emits raw blocks with the same forwarding/format contracts.
    Useful when structural row sharing already saves capacity and codec decoding
    is more expensive than its marginal byte reduction. True preserves defaults.
    max_block_bytes optionally caps decoded bytes per block, retaining at least
    one full row. Reduces sparse-miss amplification at a compression/metadata
    tradeoff; it does not bound RSS. None preserves the existing block_rows layout.
    codec='zlib' preserves the v1-v3 defaults. Explicit codec='lz4' or 'zstd'
    uses an optional system compressor and format 4 (forwarding) or 5 (identity).
    The reader must be built with the corresponding optional codec capability;
    older readers reject these versions. Compression is exact over row bytes.
    """
    source, output = Path(source), Path(output)
    block_rows = resolve_block_rows(row_bytes, block_rows, max_block_bytes)
    if source.resolve() == output.resolve() or row_bytes <= 0 or block_rows <= 0:
        raise ValueError("invalid paths or block dimensions")
    nrows, remainder = divmod(source.stat().st_size, row_bytes)
    if remainder:
        raise ValueError("partial input row")
    if predictor not in ("none", "xor"):
        raise ValueError("predictor must be none or xor")
    if codec not in ('zlib', 'lz4', 'zstd'):
        raise ValueError('codec must be zlib, lz4 or zstd')
    from .block_compression import compressor
    encode_block = compressor(codec) if compress else None
    codec_id = {'zlib': 1, 'lz4': 3, 'zstd': 5}[codec]
    if type(compress) is not bool:
        raise ValueError("compress must be boolean")
    if type(implicit_identity) is not bool:
        raise ValueError("implicit identity must be boolean")
    omit_map = implicit_identity and not deduplicate and row_order is None and representatives is None
    row_map_bytes = 0 if omit_map else nrows * ROW.size
    if representatives is not None:
        representatives = list(representatives)
        if (len(representatives) != nrows or
                any(type(i) is not int or not 0 <= i < nrows for i in representatives) or
                any(representatives[r] != r for r in representatives)):
            raise ValueError("representatives must be an idempotent map of original row IDs")
    if row_order is not None:
        row_order = list(row_order)
        if (len(row_order) != nrows or
                any(type(i) is not int or not 0 <= i < nrows for i in row_order) or
                len(set(row_order)) != nrows):
            raise ValueError("row_order must be a permutation of all original row IDs")
    unique, blocks, seen, pending, physical = 0, [], {}, [], {}
    reordered_map = bytearray(nrows * ROW.size) if row_order is not None and representatives is None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output.parent) as map_file, tempfile.TemporaryFile(dir=output.parent) as spool:
        def flush():
            raw = b"".join(pending)
            compressed = encode_block(raw) if compress else raw
            selected_codec = codec_id if len(compressed) < len(raw) else 0
            encoded = compressed if selected_codec else raw
            if compress and predictor == "xor" and len(pending) > 1:
                # Python integers perform the same bytewise operation in C.
                # A little-endian shift places the previous row at the current
                # row's position, leaving the first row unchanged. All buffers
                # remain bounded by this block; no dtype/float conversion occurs.
                residual = (int.from_bytes(raw, "little") ^
                            (int.from_bytes(raw[:-row_bytes], "little") << (8 * row_bytes)))
                residual = residual.to_bytes(len(raw), "little")
                candidate = encode_block(residual)
                if len(candidate) < len(encoded):
                    encoded, selected_codec = candidate, codec_id + 1
            blocks.append((spool.tell(), len(encoded), len(raw), selected_codec))
            spool.write(encoded)
            pending.clear()

        with source.open("rb") as stream:
            for original_id in range(nrows) if row_order is None else row_order:
                if representatives is not None and representatives[original_id] != original_id:
                    continue
                if row_order is not None or representatives is not None:
                    stream.seek(original_id * row_bytes)
                row = stream.read(row_bytes)
                if len(row) != row_bytes:
                    raise ValueError("source changed or truncated while exporting")
                index = seen.get(row) if deduplicate else None
                if index is None:
                    index = unique
                    unique += 1
                    if deduplicate:
                        seen[row] = index
                    pending.append(row)
                    if len(pending) == block_rows:
                        flush()
                if representatives is not None:
                    physical[original_id] = index
                else:
                    if reordered_map is not None:
                        ROW.pack_into(reordered_map, original_id * ROW.size, index)
                    elif not omit_map:
                        map_file.write(ROW.pack(index))
            if os.fstat(stream.fileno()).st_size != nrows * row_bytes:
                raise ValueError("source size changed while exporting")
        if pending:
            flush()
        if reordered_map is not None:
            map_file.write(reordered_map)
        if representatives is not None:
            for representative in representatives:
                map_file.write(ROW.pack(physical[representative]))
        payload_start = HEADER.size + row_map_bytes + len(blocks) * DESC.size
        with _atomic_output(output) as dest:
            version = 3 if omit_map else (2 if predictor == "xor" else 1)
            if codec != 'zlib':
                version = 5 if omit_map else 4
            dest.write(HEADER.pack(b"MEFTNAT1", version, nrows, row_bytes, unique, block_rows, len(blocks)))
            map_file.seek(0)
            for part in iter(lambda: map_file.read(1024 * 1024), b""):
                dest.write(part)
            for offset, stored, raw_size, selected_codec in blocks:
                dest.write(DESC.pack(payload_start + offset, stored, raw_size, selected_codec))
            spool.seek(0)
            for part in iter(lambda: spool.read(1024 * 1024), b""):
                dest.write(part)
    return {"format": "MEFTNAT1", "format_version": version, "input_bytes": nrows * row_bytes,
            "artifact_bytes": output.stat().st_size, "unique_rows": unique,
            "row_map_bytes": row_map_bytes, "n_rows": nrows, "row_bytes": row_bytes,
            "forwarding": "implicit-identity" if omit_map else "explicit",
            "block_rows": block_rows, "blocks": len(blocks),
            "max_block_bytes": max_block_bytes,
            "raw_blocks": sum(codec == 0 for _, _, _, codec in blocks),
            "compressed_blocks": sum(codec != 0 for _, _, _, codec in blocks),
            "zlib_blocks": sum(codec == 1 for _, _, _, codec in blocks),
            "lz4_blocks": sum(c in (3, 4) for _, _, _, c in blocks),
            "zstd_blocks": sum(c in (5, 6) for _, _, _, c in blocks),
            "xor_blocks": sum(codec in (2, 4, 6) for _, _, _, codec in blocks),
            "deduplicate": deduplicate, "predictor": predictor,
            "compression": compress, "block_codec": codec,
            "reordered": row_order is not None,
            "external_representatives": representatives is not None,
            "marked_rows": nrows if representatives is None else len(physical)}


class NativeRows:
    """Single-worker C handle. Library path is explicit or MEFT_ROWS_LIBRARY."""
    def __init__(self, path, library=None, cache_bytes=0, *, cache_policy="blocks"):
        if cache_policy not in ("blocks", "rows"):
            raise ValueError("cache policy must be blocks or rows")
        library = library or os.environ.get("MEFT_ROWS_LIBRARY")
        if not library:
            root = Path(__file__).resolve().parents[1]
            for directory in (root / "_native", root / "meft_memory" / "_native"):
                for filename in ("libmeft_rows.so", "libmeft_rows.dylib"):
                    if (directory / filename).is_file():
                        library = directory / filename
                        break
                if library:
                    break
        if not library:
            raise ValueError("set MEFT_ROWS_LIBRARY or pass a compiled library path")
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= ctypes.c_size_t(-1).value:
            raise ValueError("cache bytes must fit a nonnegative size_t")
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.meft_rows_open.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.meft_rows_open.restype = ctypes.c_void_p
        self.lib.meft_rows_close.argtypes = [ctypes.c_void_p]
        self.lib.meft_rows_close.restype = None
        for name in ("count", "width", "last_block_reads", "last_decompressions"):
            method = getattr(self.lib, "meft_rows_" + name)
            method.argtypes = [ctypes.c_void_p]
            method.restype = ctypes.c_uint64
        self.lib.meft_rows_gather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.meft_rows_gather.restype = ctypes.c_int
        error = ctypes.create_string_buffer(256)
        self.handle = self.lib.meft_rows_open(os.fsencode(path), error, len(error))
        if not self.handle:
            raise ValueError(error.value.decode())
        self.header = {"n_rows": self.lib.meft_rows_count(self.handle),
                       "row_bytes": self.lib.meft_rows_width(self.handle)}
        # Keep zero-budget loading compatible with older compiled libraries.
        self.cache_policy = "blocks"
        if cache_bytes or cache_policy != "blocks":
            try:
                if cache_bytes:
                    self.set_cache_bytes(cache_bytes)
                if cache_policy != "blocks":
                    self.set_cache_policy(cache_policy)
            except Exception:
                self.close()
                raise

    def set_cache_bytes(self, budget):
        if not self.handle:
            raise ValueError("row store is closed")
        if type(budget) is not int or not 0 <= budget <= ctypes.c_size_t(-1).value:
            raise ValueError("cache bytes must fit a nonnegative size_t")
        try:
            setter = self.lib.meft_rows_set_cache_bytes
        except AttributeError as error:
            raise ValueError("native cache requires rebuilding the row library") from error
        setter.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        setter.restype = ctypes.c_int
        if setter(self.handle, budget):
            raise ValueError("cannot configure native cache")

    @property
    def decoder_name(self):
        if not self.handle:
            raise ValueError("row store is closed")
        if not hasattr(self.lib, "meft_rows_decoder_name"):
            return "zlib"
        function = self.lib.meft_rows_decoder_name
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_char_p
        return function(self.handle).decode("ascii")

    @property
    def supports_bounded_bag_compiler(self):
        return hasattr(self.lib, 'meft_rows_compile_bounded_bags')

    def compile_bounded_bags_into(self, rows_address, counts_address, offsets_address, count, bags,
            mapping_address, nrows, sizes_address, groups, row_l2_address, total_l2_address, defect_l2_address,
            ids_address, weights_address, plan_offsets_address, projection_address, exact_address,
            plan_capacity, bag_capacity):
        """Pointer ABI; caller owns disjoint, correctly sized immutable arrays/output.

        IDs/counts are sorted unique per bag. Query scratch is bounded by the
        largest bag. No stored table bytes are read or changed during compilation.
        """
        if not self.handle:
            raise ValueError('row store is closed')
        if not self.supports_bounded_bag_compiler:
            raise ValueError('bounded bag compilation requires rebuilding the native row library')
        function = self.lib.meft_rows_compile_bounded_bags
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        function.argtypes = [pointer, pointer, pointer, size, size, pointer, size, pointer, size,
            pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, size, size,
            pointer, pointer, size]
        function.restype = ctypes.c_int
        used = ctypes.c_size_t()
        error = ctypes.create_string_buffer(256)
        status = function(rows_address, counts_address, offsets_address, count, bags,
            mapping_address, nrows, sizes_address, groups, row_l2_address, total_l2_address, defect_l2_address,
            ids_address, weights_address, plan_offsets_address, projection_address, exact_address,
            plan_capacity, bag_capacity, ctypes.byref(used), error, len(error))
        if status:
            raise ValueError(error.value.decode())
        return used.value

    def _counter(self, name):
        if not self.handle:
            raise ValueError("row store is closed")
        method = getattr(self.lib, "meft_rows_" + name)
        method.argtypes = [ctypes.c_void_p]
        method.restype = ctypes.c_uint64
        return method(self.handle)

    def set_cache_policy(self, policy):
        if not self.handle:
            raise ValueError("row store is closed")
        if policy not in ("blocks", "rows"):
            raise ValueError("cache policy must be blocks or rows")
        try:
            setter = self.lib.meft_rows_set_cache_policy
        except AttributeError as error:
            raise ValueError("row-cache policy requires rebuilding the row library") from error
        setter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        setter.restype = ctypes.c_int
        if setter(self.handle, int(policy == "rows")):
            raise ValueError("cannot configure native cache policy")
        self.cache_policy = policy

    @property
    def cache_bytes(self):
        return self._counter("cache_bytes")

    @property
    def last_cache_hits(self):
        return self._counter("last_cache_hits")

    @property
    def last_row_cache_hits(self):
        if not self.handle:
            raise ValueError("row store is closed")
        if not hasattr(self.lib, "meft_rows_last_row_cache_hits"):
            return 0  # Older block-only libraries remain supported.
        return self._counter("last_row_cache_hits")

    def gather_into(self, ids_address, count, output_address, output_bytes, *, max_query_rows=None):
        if not self.handle:
            raise ValueError("row store is closed")
        if max_query_rows is not None and (type(max_query_rows) is not int or
                                          not 1 <= max_query_rows <= ctypes.c_size_t(-1).value):
            raise ValueError("max_query_rows must be a positive size_t or None")
        chunked = max_query_rows is not None and count > max_query_rows
        if chunked:
            try:
                function = self.lib.meft_rows_gather_chunked
            except AttributeError as error:
                raise ValueError("chunked queries require rebuilding the native row library") from error
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                 ctypes.c_void_p, ctypes.c_size_t]
            function.restype = ctypes.c_int
        else:
            function = self.lib.meft_rows_gather
        error = ctypes.create_string_buffer(256)
        args = [self.handle, ids_address, count, output_address, output_bytes]
        if chunked:
            args.append(max_query_rows)
        status = function(*args, error, len(error))
        if status:
            raise ValueError(error.value.decode())

    def embedding_bag_into(self, ids_address, count, offsets_address, bags, dtype,
                           mean, padding_id, chunk_rows, output_address, output_floats,
                           *, weights_address=None, weights_bytes=0):
        """Reduce original row IDs into float32 output with bounded row staging."""
        if not self.handle:
            raise ValueError("row store is closed")
        weighted = weights_address is not None
        if weighted and mean:
            raise ValueError("per-sample weights require sum mode")
        if weighted and (type(weights_bytes) is not int or
                         not 0 <= weights_bytes <= ctypes.c_size_t(-1).value):
            raise ValueError("weights bytes must fit a nonnegative size_t")
        try:
            function = getattr(self.lib, "meft_rows_embedding_bag_weighted" if weighted
                               else "meft_rows_embedding_bag")
        except AttributeError as error:
            raise ValueError("embedding bags require rebuilding the native row library") from error
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int] + (
                             [ctypes.c_void_p, ctypes.c_size_t] if weighted else [ctypes.c_int]) + [
                             ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_size_t]
        function.restype = ctypes.c_int
        error = ctypes.create_string_buffer(256)
        arguments = [self.handle, ids_address, count, offsets_address, bags, dtype]
        arguments += [weights_address, weights_bytes] if weighted else [int(mean)]
        status = function(*arguments, padding_id, chunk_rows, output_address, output_floats,
                          error, len(error))
        if status:
            raise ValueError(error.value.decode())

    def rows(self, ids):
        ids = list(ids)
        if any(type(i) is not int or not 0 <= i < self.header["n_rows"] for i in ids):
            raise IndexError("row ID out of bounds")
        query = (ctypes.c_uint64 * len(ids))(*ids)
        output = ctypes.create_string_buffer(len(ids) * self.header["row_bytes"])
        self.gather_into(ctypes.addressof(query), len(ids), ctypes.addressof(output), len(output))
        return output.raw

    def row(self, row_id):
        return self.rows([row_id])

    @property
    def last_block_reads(self):
        return self.lib.meft_rows_last_block_reads(self.handle)

    @property
    def last_decompressions(self):
        return self.lib.meft_rows_last_decompressions(self.handle)

    @property
    def last_compressed_bytes(self):
        return self._counter("last_compressed_bytes")

    @property
    def last_decoded_bytes(self):
        return self._counter("last_decoded_bytes")

    def close(self):
        if self.handle:
            self.lib.meft_rows_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
