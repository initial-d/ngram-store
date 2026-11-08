"""Real tiny Qwen CPU interface fixtures; no compression/quality benchmark."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    import torch
    import transformers
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    if transformers.__version__ != '5.3.0':
        torch = None
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get('MEFT_COUNTERFACT_SOURCE')


@unittest.skipUnless(torch is not None and SOURCE and shutil.which('cc'),
                     'Torch, Transformers 5.3.0, pinned Engram source and C compiler required')
class CounterfactNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.library = cls.root / 'librows.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-fPIC', '-shared',
                        str(ROOT / 'native/meft_rows.c'), '-lz', '-o', str(cls.library)], check=True)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.base = cls.root / 'base'
        cls.base.mkdir()
        vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, **{f'w{i}': i + 3 for i in range(29)}}
        raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
                                            pad_token='[PAD]', eos_token='[EOS]')
        tokenizer.save_pretrained(cls.base)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=10, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=16, tie_word_embeddings=True, pad_token_id=0, eos_token_id=2)
        config._attn_implementation = 'eager'
        torch.manual_seed(83)
        Qwen3ForCausalLM(config).save_pretrained(cls.base)
        cls.prepared = cls.root / 'prepared'
        cls.prepared.mkdir()
        cls.payload = {'train': [0, 1], 'sequences': [
            {'ids': [3, 4, 5, 6, 7, 2], 'target_start': 3},
            {'ids': [3, 6, 5, 8, 2], 'target_start': 3},
            {'ids': [3, 9, 5, 2], 'target_start': 3},
            {'ids': [20, 21, 22, 2], 'target_start': 3}],
            'cases': [{'case_id': 17, 'split': 'test', 'groups': {
                'rewrite': [{'new': 0}], 'paraphrase': [{'new': 2}],
                'neighborhood': [{'new': 3}]}}]}
        cls.write(cls.prepared / 'sequences.json', cls.payload)
        from tools.counterfact_capacity import sha
        cls.write(cls.prepared / 'prepare.json', {'payload_sha256': sha(cls.prepared / 'sequences.json')})

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.tmp.cleanup()

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data, indent=2) + '\n')

    def fixture(self, variant):
        from tools.counterfact_capacity import sha
        from tools.counterfact_models import setup
        from tools.ngram_identity import Vocabulary, configure_memory
        folder = self.root / f'run-{variant}-{len(list(self.root.iterdir()))}'
        folder.mkdir()
        protocol = {'suffix_layers': 8, 'train_decoder_suffix': False,
                    'variants': {variant: {'control': 'full', 'vocab_sizes': [257, 521, 1031]}}}
        model, _, replay, module, _, _ = setup(self.base, SOURCE, 'full', [257, 521, 1031], 3, device='cpu',
                                              train_decoder_suffix=False)
        info = None
        if variant != 'legacy':
            protocol['identity_control'] = True
            protocol['variants'][variant]['identity'] = variant
            vocabulary = Vocabulary.build(self.payload, replay.memory.hash_mapping, sha(self.prepared / 'prepare.json'))
            vocabulary.save(folder / 'vocabulary.npz')
            info = configure_memory(replay, module, vocabulary, variant == 'exact', 3)
            info.update(vocabulary_sha256=sha(folder / 'vocabulary.npz'),
                        vocabulary_file_bytes=(folder / 'vocabulary.npz').stat().st_size)
        before = replay.memory.multi_head_embedding.embedding.weight.detach().clone()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        for _ in range(3):
            optimizer.zero_grad()
            replay.complete(ids, True, 1).square().mean().backward()
            optimizer.step()
        self.assertFalse(torch.equal(before, replay.memory.multi_head_embedding.embedding.weight))
        module.set_global_step(3)
        state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
        save_file(state, str(folder / 'delta.safetensors'))
        report = {'status': 'complete', 'protocol': protocol, 'seed': 3, 'variant': variant,
                  'optimizer_steps': 3, 'prepared_sha256': sha(self.prepared / 'prepare.json'),
                  'protocol_sha256': 'correctness-fixture', 'frozen_weight_sha256': replay.frozen_hash(),
                  'frozen_decoder_suffix_sha256': replay.frozen_suffix_hash(),
                  'delta': {'sha256': sha(folder / 'delta.safetensors'),
                            'bytes': (folder / 'delta.safetensors').stat().st_size}}
        if info is not None:
            report['identity'] = info
            self.write(folder / 'model-recipe.json', {'protocol': protocol, 'seed': 3, 'variant': variant,
                'delta_sha256': report['delta']['sha256'], 'identity': info,
                'vocabulary': 'vocabulary.npz', 'vocabulary_sha256': info['vocabulary_sha256']})
        self.write(folder / 'report.json', report)
        return folder, model, replay

    def export(self, run, native_oov=False):
        output = self.root / f'bundle-{len(list(self.root.iterdir()))}'
        command = [os.sys.executable, '-m', 'tools.counterfact_native', '--run', str(run),
                   '--base', str(self.base), '--upstream', str(SOURCE), '--prepared', str(self.prepared),
                   '--output', str(output), '--native-library', str(self.library), '--requests', '3', '--device', 'cpu']
        if native_oov:
            command.append('--native-oov')
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, 'OMP_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((output / 'report.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['execution_device'], 'cpu')
        self.assertTrue(report['all_original_rows_byte_exact'])
        self.assertTrue(report['independent_native_bundle_reopen_byte_exact'])
        self.assertEqual(len(report['requests']), 3)
        return output

    def test_restore_trained_exact_rows_and_unknown_ids(self):
        from tools.counterfact_native import restore, digest
        run, source, replay = self.fixture('exact')
        self.addCleanup(replay.close)
        _, restored, other, _ = restore(run, self.base, SOURCE, device='cpu')
        self.addCleanup(other.close)
        for ids in (torch.tensor([[3, 4, 5]]), torch.tensor([[20, 21, 22]])):
            self.assertEqual(digest(replay.complete(ids, True, 1)), digest(other.complete(ids, True, 1)))
        self.assertTrue(other.memory.hash_mapping.exact)
        self.assertFalse(other.memory.hash_mapping.native_oov)
        self.assertFalse(any(p.requires_grad for p in restored.parameters()))

    def test_exact_native_export_and_independent_reopen(self):
        from tools.counterfact_native import reopen, digest
        run, source, replay = self.fixture('exact')
        self.addCleanup(replay.close)
        bundle = self.export(run)
        model, native, staged = reopen(bundle, SOURCE, self.library, device='cpu')
        self.addCleanup(native.close); self.addCleanup(staged.close)
        self.assertTrue((bundle / 'vocabulary.npz').is_file())
        for ids in (torch.tensor([[3, 4, 5]]), torch.tensor([[20, 21, 22]])):
            self.assertEqual(digest(replay.complete(ids, True, 1)), digest(native.complete(ids, True, 1)))

    def test_hash_native_oov_and_legacy_bundle_compatibility(self):
        for variant, native_oov in (('hash', True), ('legacy', False)):
            with self.subTest(variant=variant):
                run, source, replay = self.fixture(variant)
                self.addCleanup(replay.close)
                bundle = self.export(run, native_oov)
                recipe = json.loads((bundle / 'model-recipe.json').read_text())
                if variant == 'hash':
                    self.assertTrue(recipe['native_oov'])
                else:
                    self.assertNotIn('vocabulary', recipe)

    def test_changed_vocabulary_and_exact_oov_mode_rejected(self):
        from tools.counterfact_native import restore
        run, source, replay = self.fixture('exact')
        self.addCleanup(replay.close)
        with self.assertRaisesRegex(ValueError, 'exact-key'):
            restore(run, self.base, SOURCE, device='cpu', native_oov=True)
        (run / 'vocabulary.npz').write_bytes(b'changed vocabulary')
        with self.assertRaisesRegex(ValueError, 'vocabulary checksum'):
            restore(run, self.base, SOURCE, device='cpu')

    def test_equal_total_rows_do_not_permit_changed_head_partition(self):
        from tools.counterfact_capacity import sha
        from tools.counterfact_native import reopen
        run, source, replay = self.fixture('exact')
        self.addCleanup(replay.close)
        bundle = self.export(run)
        path = bundle / 'memory/embedding.json'
        meta = json.loads(path.read_text())
        meta['head_sizes'][0] += 1; meta['head_sizes'][-1] -= 1
        meta['offsets'] = [sum(meta['head_sizes'][:i]) for i in range(len(meta['head_sizes']))]
        self.write(path, meta)
        recipe_path = bundle / 'model-recipe.json'
        recipe = json.loads(recipe_path.read_text()); recipe['memory_manifest_sha256'] = sha(path)
        self.write(recipe_path, recipe)
        with self.assertRaisesRegex(ValueError, 'local-head layout'):
            reopen(bundle, SOURCE, self.library, device='cpu')


if __name__ == '__main__':
    unittest.main()
