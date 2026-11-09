"""Storage benchmark preserves actual checkpoint rows and direct reopen."""
import json
from types import SimpleNamespace
import unittest

from tests import test_counterfact_native as fixture


@unittest.skipUnless(not getattr(fixture.CounterfactNativeTest, '__unittest_skip__', False),
                     'optional Qwen/TinyEngram dependencies or C compiler unavailable')
class CounterfactStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CounterfactNativeTest.setUpClass()
        cls.helper = fixture.CounterfactNativeTest()

    @classmethod
    def tearDownClass(cls):
        fixture.CounterfactNativeTest.tearDownClass()

    def test_trained_bundle_reopens_and_native_worker_never_needs_source_rows(self):
        from tools.benchmark_counterfact_storage import prepare, worker, sweep
        from tools.counterfact_native import reopen, digest
        torch, SOURCE = fixture.torch, fixture.SOURCE
        run, model, replay = self.helper.fixture('exact')
        self.addCleanup(replay.close)
        r = json.loads((run / 'report.json').read_text())
        r.update(trained_parameter_names=[n for n, p in model.named_parameters() if p.requires_grad],
                 metrics={'fixture': 'interface only'})
        self.helper.write(run / 'report.json', r)
        output = self.helper.root / 'storage-check'
        args = SimpleNamespace(run=run, prepared=self.helper.prepared, base=self.helper.base,
            upstream=SOURCE, output=output, native_library=self.helper.library,
            structural_library=None, requests=3, block_rows=32)
        prepare(args)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'passed')
        self.assertEqual(len(manifest['variants']), 3)
        self.assertEqual(manifest['unaddressed_budget_padding_rows'],
                         r['identity']['physical_rows'] - r['identity']['logical_exact_rows_with_zero_sentinels'])
        self.assertTrue(all(v['all_original_rows_byte_exact'] for v in manifest['variants']))
        sweep_output = self.helper.root / 'sweep-check'
        sweep(SimpleNamespace(fixture=output, output=sweep_output, native_library=self.helper.library,
                              block_sizes=[32, 128]))
        for block in (32, 128):
            report = sweep_output / f'query-b{block}.json'
            worker(SimpleNamespace(fixture=sweep_output, native_library=self.helper.library,
                mode=f'xor-b{block}', output=report, cache_bytes=1024, warmup=0, repeat=1))
            self.assertTrue(json.loads(report.read_text())['all_query_outputs_byte_exact'])
        for variant in manifest['variants']:
            native_model, native, staged = reopen(output / variant['name'], SOURCE, self.helper.library, device='cpu')
            try:
                for ids in (torch.tensor([[3, 4, 5]]), torch.tensor([[20, 21, 22]])):
                    self.assertEqual(digest(replay.complete(ids, True, 1)), digest(native.complete(ids, True, 1)))
            finally:
                staged.close()
                native.close()
        (output / 'rows.f32').unlink()
        for variant in manifest['variants']:
            report = output / (variant['name'] + '-worker.json')
            worker(SimpleNamespace(fixture=output, native_library=self.helper.library,
                mode=variant['name'], output=report, cache_bytes=1024, warmup=1, repeat=2))
            result = json.loads(report.read_text())
            self.assertEqual(result['status'], 'passed')
            self.assertTrue(result['all_query_outputs_byte_exact'])
            self.assertEqual(len(result['samples_ms']), 6)
        from tools.benchmark_counterfact_model import worker as model_worker
        paired = []
        for mode in ('source', 'native'):
            path = output / ('model-' + mode + '.json')
            model_worker(SimpleNamespace(run=run, base=self.helper.base, upstream=SOURCE,
                prepared=self.helper.prepared, native_library=self.helper.library,
                output=path, mode=mode, bundle=output / 'xor', device='cpu',
                requests=3, warmup=0, repeat=1))
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['memory_row_parameter_present'], mode == 'source')
            paired.append(result['requests'])
        self.assertEqual(paired[0], paired[1])


if __name__ == '__main__':
    unittest.main()
