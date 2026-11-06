"""Original-ID codec staging lifetime and head-bound tests, not compression claims."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    import torch
    import numpy as np
except ImportError:
    torch=None

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None and shutil.which('cc'),'Torch, NumPy and compiler required')
class CodecStagingTest(unittest.TestCase):
    def exercise(self,device):
        from tools.engram_row_adapter import export_multihead
        from tools.quantized_memory import pack_quantized, decode_rows, encode_rows
        from tools.codec_staged_baseline import CodecStagedBaseline
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); library=root/'librows.so'
            subprocess.run(['cc','-std=c11','-O2','-shared','-fPIC',str(ROOT/'native/meft_rows.c'),'-lz','-o',str(library)],check=True)
            weight=torch.arange(91,dtype=torch.float32).reshape(13,7)/13
            weight[0,0]=-0.0
            export_multihead(weight,[5,8],root/'source',backend='native',block_rows=3)
            ids=torch.tensor([[[0,7],[4,0],[0,7]]])
            global_ids=(ids+torch.tensor([0,5])).flatten().numpy()
            for encoding in ('fp16','int8'):
                expected=decode_rows(encode_rows(weight.numpy(),encoding),7,encoding)[global_ids].reshape(*ids.shape,7)
                for backend in ('raw','native'):
                    directory=root/(encoding+backend)
                    pack_quantized(weight.numpy(),directory,encoding,native=backend=='native')
                    with CodecStagedBaseline(root/'source',directory,device=device,native_library=library,staging_rows=2,staging_slots=2,native_cache_bytes=128) as adapter:
                        tickets=[adapter.prefetch(ids) for _ in range(3)]
                        for ticket in tickets:
                            np.testing.assert_array_equal(ticket.wait().cpu().numpy().view('u1'),expected.view('u1'))
                        self.assertLessEqual(adapter.stats['staging_bytes'],adapter.stats['staging_byte_limit'])
                        with self.assertRaises(IndexError):adapter(torch.tensor([[5,0]]))
                    np.testing.assert_array_equal(tickets[0].wait().cpu().numpy(),expected)
                    adapter.close()
                    with self.assertRaises(ValueError):adapter(ids)

    def test_cpu_codec_original_ids_duplicates_and_closed_ticket_lifetime(self):
        self.exercise('cpu')

    def test_faiss_selected_rows_keep_head_offsets_and_duplicate_order(self):
        try:
            import faiss
        except ImportError:
            self.skipTest('optional FAISS required')
        from tools.engram_row_adapter import export_multihead
        from tools.faiss_row_codec import FaissRows,fit_codec
        from tools.codec_staged_baseline import CodecStagedBaseline
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);library=root/'librows.so'
            subprocess.run(['cc','-std=c11','-O2','-shared','-fPIC',str(ROOT/'native/meft_rows.c'),'-lz','-o',str(library)],check=True)
            rows=np.arange(91,dtype=np.float32).reshape(13,7)/13
            export_multihead(torch.from_numpy(rows),[5,8],root/'source',backend='native')
            faiss.omp_set_num_threads(1)
            fit_codec(root/'codec',rows,'sq8',rows)
            reader=FaissRows(root/'codec')
            ids=torch.tensor([[[0,7],[4,0],[0,7]]])
            expected=reader.rows((ids+torch.tensor([0,5])).flatten().numpy()).reshape(*ids.shape,7)
            with CodecStagedBaseline(root/'source',root/'codec',device='cpu',native_library=library,staging_rows=2,staging_slots=2) as adapter:
                np.testing.assert_array_equal(adapter(ids).numpy(),expected)
            reader.codes._mmap.close()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'real CUDA required')
    def test_cuda_codec_original_ids_and_staging_slot_reuse(self):
        self.exercise('cuda')
