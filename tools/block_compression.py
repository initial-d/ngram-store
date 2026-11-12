"""Optional system-library block compressors; no Python package dependency.

These generic byte compressors have no knowledge of n-grams, vectors or keys.
Native decoding requires the corresponding explicit build capability.
"""
import ctypes
import ctypes.util
from functools import lru_cache
import zlib
import struct


def encode_row_packets(raw, width):
    """Fixed-anchor XOR byte masks with independent row access, no rounding.

    Header: offset width (2/4), three reserved zeros, first original row,
    rows+1 little-endian offsets, then per-row masks and nonzero XOR bytes.
    The anchor row has an empty packet. Raw fallback is chosen by the exporter.
    """
    if width <= 0 or not raw or len(raw) % width:
        raise ValueError('complete nonempty fixed-width rows required')
    anchor = raw[:width]; masks = (width + 7) // 8
    packets = []; offsets = [0, 0]; total = 0
    for start in range(width, len(raw), width):
        row = raw[start:start + width]; mask = bytearray(masks); values = bytearray()
        for i, (a, b) in enumerate(zip(anchor, row)):
            delta = a ^ b
            if delta: mask[i // 8] |= 1 << (i % 8); values.append(delta)
        packet = bytes(mask) + values; packets.append(packet)
        total += len(packet); offsets.append(total)
    if total > (1 << 32) - 1: raise ValueError('row packet block exceeds uint32 offsets')
    offset_width = 2 if total <= 65535 else 4
    layout = '<' + ('H' if offset_width == 2 else 'I') * len(offsets)
    return bytes([offset_width, 0, 0, 0]) + anchor + struct.pack(layout, *offsets) + b''.join(packets)


@lru_cache(maxsize=None)
def compressor(codec):
    if codec not in ('zlib', 'lz4', 'zstd'):
        raise ValueError('codec must be zlib, lz4 or zstd')
    if codec == 'zlib':
        return zlib.compress
    filename = ctypes.util.find_library('lz4' if codec == 'lz4' else 'zstd')
    if not filename:
        raise ValueError('optional system ' + codec + ' library is unavailable')
    lib = ctypes.CDLL(filename)
    if codec == 'lz4':
        lib.LZ4_compressBound.argtypes = [ctypes.c_int]
        lib.LZ4_compressBound.restype = ctypes.c_int
        lib.LZ4_compress_default.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.LZ4_compress_default.restype = ctypes.c_int
        def encode(raw):
            if len(raw) > (1 << 31) - 1:
                raise ValueError('LZ4 block exceeds int32 input limit')
            bound = lib.LZ4_compressBound(len(raw))
            if bound <= 0:
                raise ValueError('LZ4 block exceeds codec input limit')
            out = ctypes.create_string_buffer(bound)
            count = lib.LZ4_compress_default(raw, out, len(raw), bound)
            if count <= 0:
                raise ValueError('LZ4 compression failed')
            return out.raw[:count]
        return encode
    if codec != 'zstd':
        raise ValueError('codec must be zlib, lz4 or zstd')
    lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.ZSTD_compress.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    def encode(raw):
        out = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(raw)))
        count = lib.ZSTD_compress(out, len(out), raw, len(raw), 3)
        if lib.ZSTD_isError(count):
            raise ValueError('Zstd compression failed')
        return out.raw[:count]
    return encode
