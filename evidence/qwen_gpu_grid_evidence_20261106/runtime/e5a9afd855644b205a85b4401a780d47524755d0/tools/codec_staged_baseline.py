"""Benchmark codec reader sharing native host-ID and CUDA staging schedule."""
from pathlib import Path
import torch
from .device_row_adapter import StagedMultiHeadEmbedding
from .row_codec_utils import close_reader, read_rows
from .faiss_row_codec import FaissRows
from .quantized_memory import QuantizedRows


class CodecStagedBaseline(StagedMultiHeadEmbedding):
    def __init__(self, native_directory, codec_directory, **kwargs):
        super().__init__(native_directory, **kwargs)
        self.gather.close()  # Keep validated original head layout only.
        directory=Path(codec_directory)
        self.reader=(FaissRows(directory) if (directory/'codes.npy').exists() else
                     QuantizedRows(directory,kwargs.get('native_library'),cache_bytes=kwargs.get('native_cache_bytes',0)))
        metadata=self.reader.metadata
        shape=(metadata['rows'],metadata['dim']) if isinstance(self.reader,FaissRows) else metadata['shape']
        if list(shape)!=self.gather.metadata['shape'] or self.dtype!=torch.float32:
            close_reader(self.reader)
            super().close()
            raise ValueError('codec shape differs from source layout')

    def _read(self,ids,output):
        values=read_rows(self.reader,ids.numpy())
        output.reshape(-1,self.embedding_dim).copy_(torch.from_numpy(values))

    def close(self):
        if not self._closed:
            super().close()
            close_reader(self.reader)
