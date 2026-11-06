"""Original-ID row codec helpers, without model/framework loading."""
import numpy as np
from .quantized_memory import QuantizedRows


def close_reader(reader):
    if isinstance(reader, QuantizedRows):
        reader.close()
    else:
        reader.codes._mmap.close()


def read_rows(reader, ids):
    return reader.gather(ids) if isinstance(reader, QuantizedRows) else reader.rows(ids)


def decode_complete(reader, shape):
    result = np.empty(shape, dtype='<f4')
    for start in range(0, shape[0], 4096):
        stop = min(start+4096, shape[0])
        result[start:stop] = read_rows(reader, np.arange(start,stop))
    return result
