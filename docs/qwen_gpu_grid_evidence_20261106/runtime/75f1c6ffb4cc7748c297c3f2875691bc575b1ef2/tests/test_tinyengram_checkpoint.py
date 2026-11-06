"""Real Qwen3 classes/local tokenizer; tiny trained fixtures test interfaces only."""
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
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config
    if transformers.__version__ != "4.57.3":
        torch = None
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("MEFT_TINYENGRAM_SOURCE")


@unittest.skipUnless(torch is not None and SOURCE and shutil.which("cc"),
                     "pinned TinyEngram source, transformers 4.57.3, PyTorch and C compiler required")
class TinyEngramCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.tinyengram_checkpoint import load_upstream
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.library = cls.root / "librows.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                        str(ROOT / "native/meft_rows.c"), "-lz", "-o", str(cls.library)], check=True)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.checkpoint = cls.root / "checkpoint"
        cls.checkpoint.mkdir()
        vocab = {"[PAD]": 0, "[UNK]": 1, **{f"word{i}": i + 2 for i in range(30)}}
        raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]")
        tokenizer.save_pretrained(cls.checkpoint)
        torch.manual_seed(731)
        engram, qwen = load_upstream(SOURCE)
        cls.engram_config = engram.EngramConfig(
            tokenizer_name_or_path=str(cls.checkpoint), engram_vocab_size=[17, 19, 23],
            max_ngram_size=4, n_embed_per_ngram=8, n_head_per_ngram=2,
            layer_ids=[0, 2], pad_id=0, kernel_size=3,
        )
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                             tie_word_embeddings=True, pad_token_id=0)
        config._attn_implementation = "eager"
        cls.original = qwen.EngramQwenForCausalLM(config, engram_cfg=cls.engram_config)
        for name, parameter in cls.original.named_parameters():
            parameter.requires_grad_("engram_layers" in name)
        cls.ids = torch.tensor([[2, 5, 3, 7, 6, 2, 4, 8, 12, 3, 9, 4, 6, 7, 2, 11, 3, 14, 8],
                                [4, 8, 3, 2, 9, 5, 2, 4, 7, 3, 8, 2, 9, 12, 2, 4, 5, 6, 7]])
        # Learn nonzero convolution weights, rather than let zero initialization
        # hide missing decode history. This tiny fixture is not a quality study.
        optimizer = torch.optim.Adam([p for p in cls.original.parameters() if p.requires_grad], lr=0.01)
        cls.original.train()
        for _ in range(4):
            optimizer.zero_grad()
            loss = cls.original(cls.ids, labels=cls.ids, use_cache=False).loss
            loss.backward()
            optimizer.step()
        cls.original.eval()
        cls.original.save_pretrained(cls.checkpoint, safe_serialization=True)
        (cls.checkpoint / "engram_config.json").write_text(json.dumps(cls.engram_config.to_dict()))
        cls.expected = cls.original(cls.ids, use_cache=False).logits.detach()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.tmp.cleanup()

    def export(self, checkpoint=None, **kwargs):
        from tools.tinyengram_checkpoint import export_tinyengram_checkpoint
        target = self.root / ("bundle-" + str(len(list(self.root.glob("bundle-*")))))
        export_tinyengram_checkpoint(checkpoint or self.checkpoint, target, upstream_dir=SOURCE,
                                    block_rows=3, shard_bytes=2048, **kwargs)
        return target

    def load(self, path, **kwargs):
        from tools.tinyengram_checkpoint import load_tinyengram_checkpoint
        return load_tinyengram_checkpoint(path, upstream_dir=SOURCE, native_library=self.library,
                                         native_cache_bytes=2048, query_chunk_rows=3, **kwargs)

    def test_complete_quality_oracle_on_trained_fixture(self):
        import hashlib
        # This trained miniature fixture checks the evaluator plumbing, not
        # the usefulness of the real Qwen branch or compression quality.
        checkpoint = self.root / "quality-checkpoint"
        shutil.copytree(self.checkpoint, checkpoint)
        (checkpoint / "trainer_state.json").write_text('{"global_step": 0}')
        corpus = self.root / "quality-corpus.txt"
        corpus.write_text(" " * 16384 + " ".join(f"word{i % 20}" for i in range(80)))
        protocol = self.root / "quality-protocol.json"
        protocol.write_text(json.dumps({
            "schema_version": 1, "excluded_prefix_characters": 16384,
            "window_tokens": 8, "calibration_windows": 2, "test_windows": 2,
            "threads": 1, "minimum_ppl_effect": .01,
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        }))
        report = self.root / "quality-report.json"
        subprocess.run([
            os.sys.executable, "-m", "tools.evaluate_tinyengram_quality",
            "--checkpoint", str(checkpoint), "--bundle", str(self.export(checkpoint)),
            "--corpus", str(corpus), "--upstream-dir", str(SOURCE),
            "--native-library", str(self.library), "--protocol", str(protocol),
            "--report-out", str(report),
        ], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = json.loads(report.read_text())
        self.assertEqual(result["status"], "passed")
        for split in ("calibration", "test"):
            self.assertEqual(result["summary"][split]["native-fp32"]["targets"], 14)
            self.assertTrue(result["summary"][split]["native-fp32"]["all_logits_byte_exact"])
            for i in range(2):
                source = result["windows"]["memory-off"][split][i]["source_nll"]
                for name in ("head-mean", "native-fp32"):
                    self.assertEqual(result["windows"][name][split][i]["source_nll"], source)

    def test_frozen_prefix_preserves_logits_gradients_and_optimizer_update(self):
        import copy
        import numpy as np
        from tools.frozen_tinyengram_prefix import FrozenTinyEngramPrefix, check_reuse
        model = copy.deepcopy(self.original)
        del model.model.engram_layers["0"]
        reuse = FrozenTinyEngramPrefix(model)
        fingerprint = reuse.frozen_weight_sha256()
        result = check_reuse(model, reuse, self.ids, gradients=True)
        self.assertTrue(result["logits_byte_exact"])
        self.assertTrue(result["branch_gradients_exact"])
        states = reuse.capture(self.ids)
        # Exercise serialization and the real optimizer, without reusing an
        # inference-mode tensor or silently quantizing cached hidden states.
        states = torch.from_numpy(np.array(states.numpy(), copy=True))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        logits = reuse.forward(self.ids,states)
        loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,32),self.ids[:,1:].reshape(-1))
        loss.backward()
        optimizer.step()
        self.assertEqual(reuse.frozen_weight_sha256(),fingerprint)
        self.assertTrue(check_reuse(model,reuse,self.ids,gradients=True)["branch_gradients_exact"])
        model.model.embed_tokens.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            FrozenTinyEngramPrefix(model)

    def test_training_windows_remain_nonoverlapping_and_inside_training_segment(self):
        from tools.train_tinyengram_identity import sample_windows
        ids = torch.arange(137).unsqueeze(0)
        windows,indices = sample_windows(ids,8,7)
        self.assertEqual(tuple(windows.shape),(7,8))
        self.assertEqual(len(set(indices.tolist())),7)
        for i,window in zip(indices,windows):
            self.assertTrue(torch.equal(window,ids[0,i*8:(i+1)*8]))
        self.assertLess(int(windows.max()),len(ids[0]))
        with self.assertRaises(ValueError):
            sample_windows(ids,8,18)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "real CUDA required")
    def test_cuda_frozen_prefix_preserves_full_logits_and_gradients(self):
        import copy
        from tools.frozen_tinyengram_prefix import FrozenTinyEngramPrefix, check_reuse
        model = copy.deepcopy(self.original)
        del model.model.engram_layers["0"]
        model.to("cuda")
        old = torch.are_deterministic_algorithms_enabled()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            reuse = FrozenTinyEngramPrefix(model)
            result = check_reuse(model,reuse,self.ids,gradients=True)
            self.assertTrue(result["logits_byte_exact"])
            self.assertTrue(result["branch_gradients_exact"])
            self.assertEqual(reuse.capture(self.ids).device.type,"cuda")
        finally:
            torch.use_deterministic_algorithms(old)
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def check_staged_model(self, device, cache_policy="blocks"):
        import copy
        from tools.tinyengram_checkpoint import TinyEngramModel
        from tools.device_row_adapter import StagedMultiHeadEmbedding
        resident = TinyEngramModel(copy.deepcopy(self.original).to(device)).eval()
        with self.load(self.export(), device=device, prefetch_memory=True,
                       staging_rows=3, staging_slots=2, native_cache_policy=cache_policy) as mapped, torch.inference_mode():
            reference = resident(self.ids).logits
            self.assertTrue(torch.equal(mapped(self.ids).logits.cpu().view(torch.uint8),
                                        reference.cpu().view(torch.uint8)))
            left, right = None, None
            for start, end in ((0, 5), (5, 8), (8, 19)):
                a = mapped(self.ids[:, start:end], past_key_values=left, use_cache=True)
                b = resident(self.ids[:, start:end], past_key_values=right, use_cache=True)
                self.assertTrue(torch.equal(a.logits.cpu().view(torch.uint8), b.logits.cpu().view(torch.uint8)))
                left, right = a.past_key_values, b.past_key_values
            self.assertEqual(left.token_tail.device.type, "cpu")
            self.assertTrue(all(t.device.type == device for t in left.conv_tails.values()))
            self.assertEqual(left.kv.layers[0].keys.device.type, device)
            for layer in mapped.model.model.engram_layers.values():
                self.assertIsInstance(layer.multi_head_embedding, StagedMultiHeadEmbedding)
                self.assertEqual(layer.multi_head_embedding.gather.offsets.device.type, "cpu")
                self.assertEqual(layer.multi_head_embedding.stats["queries"], 4)
            a = mapped.generate(self.ids[:, :6], max_new_tokens=4)
            b = resident.generate(self.ids[:, :6], max_new_tokens=4)
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.device.type, "cpu")

    def test_cpu_row_cache_full_and_cached_logits(self):
        self.check_staged_model("cpu", "rows")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "real CUDA required")
    def test_cuda_row_cache_full_and_cached_logits(self):
        self.check_staged_model("cuda", "rows")

    def test_staged_cpu_model_prefetch_preserves_full_and_cached_logits(self):
        self.check_staged_model("cpu")

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(), "MPS unavailable")
    def test_staged_mps_backbone_host_hashes_and_device_decode_history(self):
        self.check_staged_model("mps")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "real CUDA required")
    def test_staged_cuda_backbone_host_hashes_and_device_decode_history(self):
        self.check_staged_model("cuda")

    def test_complete_forward_rows_and_tied_backbone_survive_without_source_checkpoint(self):
        bundle = self.export()
        with self.load(bundle) as mapped, torch.inference_mode():
            self.assertTrue(torch.equal(mapped(self.ids).logits, self.expected))
            self.assertIs(mapped.model.lm_head.weight, mapped.model.model.embed_tokens.weight)
            self.assertFalse(any(p.is_meta for p in mapped.parameters()))
            self.assertFalse(any(p.requires_grad for p in mapped.parameters()))
            for layer_id, layer in mapped.model.model.engram_layers.items():
                source = self.original.model.engram_layers[layer_id].multi_head_embedding.embedding.weight
                self.assertEqual(layer.multi_head_embedding.table.rows(range(len(source))),
                                 source.detach().contiguous().view(torch.uint8).numpy().tobytes())
                self.assertFalse(hasattr(layer.multi_head_embedding, "embedding"))
            # Tokenizer path stored in bundle is relative, not the checkpoint.
            self.assertEqual(json.loads((bundle / "engram_config.json").read_text())["tokenizer_name_or_path"],
                             "tokenizer")
            self.assertFalse(any("multi_head_embedding.embedding.weight" in key
                                 for key in json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]))
        relocated = self.root / "relocated"
        shutil.copytree(bundle, relocated)
        with self.load(relocated) as moved, torch.inference_mode():
            self.assertTrue(torch.equal(moved(self.ids).logits, self.expected))

    def test_cached_chunks_and_greedy_generation_preserve_hash_and_nonzero_conv_history(self):
        from tools.tinyengram_checkpoint import TinyEngramModel
        with self.load(self.export()) as mapped, torch.inference_mode():
            for sizes in ([1] * self.ids.shape[1], [5, 3, 11]):
                cache, outputs, start = None, [], 0
                for size in sizes:
                    output = mapped(self.ids[:, start:start + size], past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    outputs.append(output.logits)
                    start += size
                torch.testing.assert_close(torch.cat(outputs, dim=1), self.expected, rtol=2e-5, atol=2e-6)
                self.assertEqual(cache.seen_tokens, self.ids.shape[1])
                self.assertEqual(cache.token_tail.shape[1], 3)
                self.assertEqual(set(cache.conv_tails), {"0", "2"})
                self.assertTrue(all(t.shape[1] == 8 for t in cache.conv_tails.values()))
            resident = TinyEngramModel(self.original).eval()
            try:
                prompt = self.ids[:, :6]
                a = mapped.generate(prompt, max_new_tokens=7, use_cache=True)
                b = mapped.generate(prompt, max_new_tokens=7, use_cache=False)
                c = resident.generate(prompt, max_new_tokens=7, use_cache=True)
                self.assertTrue(torch.equal(a, b) and torch.equal(a, c))
            finally:
                # Restore the shared resident test model's methods for other tests.
                for layer in self.original.model.engram_layers.values():
                    layer.hash_mapping = layer.hash_mapping.original
                    del layer.short_conv.forward

    def test_byte_capped_blocks_preserve_full_and_cached_logits(self):
        from tools.tinyengram_checkpoint import TinyEngramModel
        bundle = self.export(max_block_bytes=32)
        metadata_files = list(bundle.glob('memory/*/embedding.json'))
        self.assertEqual(len(metadata_files), 2)
        for metadata in metadata_files:
            layout = json.loads(metadata.read_text())
            width = layout['shape'][1] * 4  # This fixture uses float32.
            self.assertEqual(layout['max_block_bytes'], 32)
            self.assertEqual(layout['block_rows'], min(3, max(1, 32 // width)))
        with self.load(bundle) as mapped, torch.inference_mode():
            self.assertTrue(torch.equal(mapped(self.ids).logits, self.expected))
            resident = TinyEngramModel(self.original).eval()
            try:
                left, right = None, None
                for start, end in ((0, 5), (5, 8), (8, 19)):
                    a = mapped(self.ids[:, start:end], past_key_values=left, use_cache=True)
                    b = resident(self.ids[:, start:end], past_key_values=right, use_cache=True)
                    self.assertTrue(torch.equal(a.logits, b.logits))
                    left, right = a.past_key_values, b.past_key_values
            finally:
                for layer in self.original.model.engram_layers.values():
                    layer.hash_mapping = layer.hash_mapping.original
                    del layer.short_conv.forward

    def test_independent_requests_and_invalid_cache_inputs(self):
        with self.load(self.export()) as mapped, self.load(self.export()) as other, torch.inference_mode():
            left = mapped(self.ids[:1, :4], use_cache=True).past_key_values
            right = mapped(self.ids[1:, :4], use_cache=True).past_key_values
            for cache, tokens in ((left, self.ids[:1, 4:8]), (right, self.ids[1:, 4:8])):
                output = mapped(tokens, past_key_values=cache, use_cache=True)
                index = 0 if cache is left else 1
                torch.testing.assert_close(output.logits, self.expected[index:index + 1, 4:8], rtol=2e-5, atol=2e-6)
            with self.assertRaisesRegex(ValueError, "different model"):
                other(self.ids[:1, 8:9], past_key_values=left, use_cache=True)
            from transformers.cache_utils import DynamicCache
            with self.assertRaisesRegex(ValueError, "valid TinyEngramCache"):
                mapped(self.ids, past_key_values=DynamicCache(config=mapped.config), use_cache=True)
            with self.assertRaisesRegex(ValueError, "batch size"):
                mapped(self.ids[:, 8:9], past_key_values=left, use_cache=True)
            with self.assertRaisesRegex(ValueError, "unpadded"):
                mapped(self.ids, attention_mask=torch.zeros_like(self.ids))
            with self.assertRaisesRegex(ValueError, "requires use_cache"):
                mapped(self.ids[:1, 8:9], past_key_values=left)
            with self.assertRaisesRegex(ValueError, "return_dict"):
                mapped(self.ids, return_dict=False)
            with self.assertRaises(IndexError):
                mapped(torch.full((1, 1), 1000), past_key_values=left, use_cache=True)
            self.assertFalse(left.valid)
            with self.assertRaisesRegex(ValueError, "valid TinyEngramCache"):
                mapped(self.ids[:1, 8:9], past_key_values=left, use_cache=True)

    def test_indexed_input_shards_preserve_full_model(self):
        source = self.root / "sharded"
        shutil.copytree(self.checkpoint, source)
        with safe_open(source / "model.safetensors", framework="pt") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        weight_map = {}
        for index, group in enumerate((list(state)[::2], list(state)[1::2])):
            name = f"part-{index}.safetensors"
            save_file({key: state[key] for key in group}, source / name)
            weight_map.update({key: name for key in group})
        (source / "model.safetensors").unlink()
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        with self.load(self.export(source)) as mapped, torch.inference_mode():
            self.assertTrue(torch.equal(mapped(self.ids).logits, self.expected))

    def test_incomplete_checkpoint_fails_before_publishing_bundle(self):
        source = self.root / "incomplete"
        shutil.copytree(self.checkpoint, source)
        with safe_open(source / "model.safetensors", framework="pt") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        del state["model.engram_layers.0.value_proj.weight"]
        save_file(state, source / "model.safetensors")
        before = set(self.root.glob("bundle-*"))
        with self.assertRaisesRegex(ValueError, "complete backbone"):
            self.export(source)
        self.assertEqual(before, set(self.root.glob("bundle-*")))
        self.assertFalse(list(self.root.glob(".meft-tiny-*")))

    def test_bundle_corruption_and_unknown_config_are_rejected(self):
        bundle = self.export()
        with (bundle / "backbone-00001.safetensors").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load(bundle)
        with self.assertRaisesRegex(ValueError, "unknown Engram"):
            self.export(engram_config={**self.engram_config.to_dict(), "misspelled_order": 4})

    def test_gate_warmup_requires_checkpoint_step_metadata(self):
        values = {**self.engram_config.to_dict(), "warmup_steps": 10}
        with self.assertRaisesRegex(ValueError, "global_step"):
            self.export(engram_config=values)


if __name__ == "__main__":
    unittest.main()
