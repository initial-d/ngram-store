"""Actual Qwen forwards with codec rows; fixtures are not task evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import unittest

from tests import test_counterfact_native as fixture


@unittest.skipUnless(not getattr(fixture.CounterfactNativeTest, '__unittest_skip__', False)
                     and importlib.util.find_spec('faiss'), 'Qwen fixture and FAISS required')
class LlmCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CounterfactNativeTest.setUpClass()
        cls.helper = fixture.CounterfactNativeTest()

    @classmethod
    def tearDownClass(cls): fixture.CounterfactNativeTest.tearDownClass()

    def test_raw_native_and_faiss_reopen_without_dense_row_parameter(self):
        import faiss
        import numpy as np
        from safetensors.torch import load_file, save_file
        from tools.counterfact_native import digest as tensor_digest
        from tools.faiss_row_codec import fit_codec, pack_codec_native
        from tools.llm_codec_frontier import reopen_codec
        from tools.quantized_memory import pack_quantized
        from tools.tree_node_quality import Reader, array_digest, digest, write
        torch = fixture.torch; faiss.omp_set_num_threads(1)
        run, model, source = self.helper.fixture('exact'); self.addCleanup(source.close)
        r = json.loads((run / 'report.json').read_text())
        common = self.helper.root / 'codec-common'; common.mkdir()
        shutil.copyfile(run / 'vocabulary.npz', common / 'vocabulary.npz')
        state = load_file(str(run / 'delta.safetensors'), device='cpu')
        rows = state.pop('ngram_memory.multi_head_embedding.embedding.weight').numpy()
        save_file(state, str(common / 'nonrow-delta.safetensors'))
        recipe = json.loads((run / 'model-recipe.json').read_text())
        recipe.update(base_model=str(self.helper.base), nonrow_delta='nonrow-delta.safetensors',
            nonrow_delta_sha256=digest(common / 'nonrow-delta.safetensors'),
            prepared_sha256=r['prepared_sha256'], global_step=3,
            frozen_weight_sha256=r['frozen_weight_sha256'],
            frozen_decoder_suffix_sha256=r['frozen_decoder_suffix_sha256'])
        write(common / 'model-recipe.json', recipe)
        write(common / 'metadata.json', {'shape': list(rows.shape), 'head_sizes': r['identity']['head_sizes']})
        for backend, encoding in (('raw', 'fp32'), ('native', 'fp32'), ('faiss', 'sq4'), ('faiss-native', 'sq4')):
            with self.subTest(backend=backend):
                directory = self.helper.root / ('codec-' + backend)
                if backend in ('raw', 'native'):
                    pack_quantized(rows, directory, encoding, native=backend == 'native')
                elif backend == 'faiss': fit_codec(directory, rows, encoding, rows, iterations=2)
                else: pack_codec_native(self.helper.root / 'codec-faiss', directory, native_library=self.helper.library)
                job = {'backend': 'faiss' if backend == 'faiss-native' else backend, 'directory': str(directory),
                    'native_library': str(self.helper.library), 'common': str(common)}
                reader = Reader(job)
                try: decoded = reader.rows(np.arange(len(rows)))
                finally: reader.close()
                effective = decoded.copy()
                effective[source.memory.multi_head_embedding.offsets.cpu().numpy()] = 0.
                with torch.no_grad(): source.memory.multi_head_embedding.embedding.weight.copy_(torch.from_numpy(effective))
                actual_model, actual, staged = reopen_codec(job, Path(fixture.SOURCE), device='cpu')
                try:
                    self.assertFalse(any('multi_head_embedding' in n for n, _ in actual_model.named_parameters()))
                    for ids in (torch.tensor([[3, 4, 5]]), torch.tensor([[20, 21, 22]])):
                        self.assertEqual(tensor_digest(source.complete(ids, True, 1)), tensor_digest(actual.complete(ids, True, 1)))
                finally: staged.close(); actual.close()
                if backend == 'raw':
                    # Complete CPU addressing uses the raw tokenizer pad ID,
                    # then the pinned compressed-token lookup. Include both
                    # official split labels in this tiny interface fixture.
                    from types import SimpleNamespace
                    from tools.benchmark_llm_codec_serving import cpu_worker
                    payload = json.loads((self.helper.prepared / 'sequences.json').read_text())
                    payload['cases'].append({**payload['cases'][0], 'split': 'calibration'})
                    write(self.helper.prepared / 'sequences.json', payload)
                    write(self.helper.prepared / 'prepare.json', {'payload_sha256': digest(self.helper.prepared / 'sequences.json')})
                    write(common / 'metadata.json', {'shape': list(rows.shape), 'head_sizes': r['identity']['head_sizes'],
                        'prepared': str(self.helper.prepared), 'base_bytes': 0})
                    report = directory / 'build-result.json'
                    write(report, {'decoded_row_sha256': array_digest(decoded), 'model_bytes': 0})
                    job['report'] = str(report)
                    job_path = common / 'cpu-job.json'; write(job_path, job)
                    output = common / 'cpu-result.json'
                    cpu_worker(SimpleNamespace(job=job_path, upstream=Path(fixture.SOURCE), cache_bytes=0,
                        requests=1, warmup=0, repeat=1, output=output))
                    self.assertEqual(json.loads(output.read_text())['status'], 'passed')


if __name__ == '__main__': unittest.main()
