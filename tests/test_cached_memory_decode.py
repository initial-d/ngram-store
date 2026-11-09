"""Causal n-gram and convolution history with both real decoder cache types."""
from unittest.mock import patch
import unittest

from tests import test_counterfact_native as fixture


@unittest.skipUnless(not getattr(fixture.CounterfactNativeTest, '__unittest_skip__', False),
                     'Qwen/TinyEngram fixture required')
class CachedDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CounterfactNativeTest.setUpClass()
        cls.helper = fixture.CounterfactNativeTest()

    @classmethod
    def tearDownClass(cls): fixture.CounterfactNativeTest.tearDownClass()

    def test_qwen3_and_qwen35_cached_outputs_match_complete_causal_forwards(self):
        torch = fixture.torch
        from tools.cached_memory_decode import generate_cached
        from tools.late_memory_replay import LateMemoryReplay
        for kind in ('qwen3', 'qwen35'):
            with self.subTest(kind=kind):
                _, model, replay = self.helper.fixture('exact')
                if kind == 'qwen35':
                    memory = replay.memory; replay.close()
                    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
                    from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
                    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                        linear_value_head_dim=16, linear_conv_kernel_dim=4,
                        layer_types=['linear_attention'] * 3 + ['full_attention'],
                        rope_parameters={'rope_type':'default', 'rope_theta':10000., 'partial_rotary_factor':.5,
                            'mrope_section':[1,1,2], 'mrope_interleaved':True})
                    config._attn_implementation = 'eager'
                    with patch.object(impl, 'FusedRMSNormGated', None), patch.object(impl, 'chunk_gated_delta_rule', None):
                        model = Qwen3_5ForCausalLM(config).eval()
                    model.requires_grad_(False); memory.requires_grad_(False)
                    memory.layer_id = 2
                    memory.hash_mapping.vocab_size_across_layers[2] = memory.hash_mapping.vocab_size_across_layers.pop(replay.layer_id)
                    memory.hash_mapping.layer_ids = [2]
                    memory.hash_mapping.source.layer_ids = [2]
                    memory.hash_mapping.source.vocab_size_across_layers[2] = memory.hash_mapping.source.vocab_size_across_layers.pop(replay.layer_id)
                    replay = LateMemoryReplay(model, memory, 2)
                try:
                    model.requires_grad_(False)
                    with torch.no_grad():
                        torch.manual_seed(97)
                        replay.memory.short_conv.conv.weight.normal_(std=.1)
                        for length in (1, 3, 11):
                            prefix = torch.tensor([[3, 4, 5, 6, 7, 3, 6, 5, 8, 4, 5][:length]])
                            continuation = torch.tensor([[6, 7, 4, 5, 8, 3, 4, 5, 6, 7, 8, 9]])
                            _, outputs = generate_cached(replay, prefix, steps=12, global_step=3, continuation=continuation)
                            for step, actual in enumerate(outputs):
                                full = torch.cat((prefix, continuation[:, :step]), dim=1)
                                expected = replay.complete(full, True, 1).cpu()
                                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                finally: replay.close()


if __name__ == '__main__': unittest.main()
