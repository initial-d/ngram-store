import unittest
from unittest.mock import patch
try:
    import torch,transformers
    from transformers import Qwen3Config,Qwen3ForCausalLM
    if transformers.__version__!='5.3.0':torch=None
except ImportError:torch=None


@unittest.skipUnless(torch is not None,'Transformers 5.3.0 and Torch required')
class LateReplayTest(unittest.TestCase):
    def check(self,kind,device):
        from tools.late_memory_replay import LateMemoryReplay,DenseMemoryAdapter,check_masked_replay,masked_loss
        torch.set_num_threads(1);torch.manual_seed(73)
        if kind=='qwen3':
            c=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                num_attention_heads=2,num_key_value_heads=2,head_dim=16,tie_word_embeddings=True)
            c._attn_implementation='eager';model=Qwen3ForCausalLM(c)
        else:
            from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
            c=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                num_attention_heads=2,num_key_value_heads=2,head_dim=16,tie_word_embeddings=True,
                linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,
                linear_conv_kernel_dim=4,layer_types=['linear_attention']*3+['full_attention'],
                rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,
                                 'mrope_section':[1,1,2],'mrope_interleaved':True})
            c._attn_implementation='eager'
            if device=='cpu':
                # Installed FLA dispatch is CUDA-only; use the stock CPU fallback.
                from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
                with patch.object(impl,'FusedRMSNormGated',None),patch.object(impl,'chunk_gated_delta_rule',None):
                    model=Qwen3_5ForCausalLM(c)
            else:model=Qwen3_5ForCausalLM(c)
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(('model.layers.2.','model.layers.3.')))
        model.to(device).eval();memory=DenseMemoryAdapter(32,1024).to(device)
        replay=LateMemoryReplay(model,memory,2)
        ids=torch.tensor([[2,5,3,7,6,8,12,3],[4,8,3,2,9,5,2,4]])
        fingerprint=replay.frozen_hash()
        for enabled in (True,False):
            self.assertTrue(check_masked_replay(replay,ids,[5,4],enabled)['all_masked_gradients_exact'])
            self.assertTrue(check_masked_replay(replay,ids,[5,4],enabled,logits_to_keep=5)['all_masked_gradients_exact'])
        replay.discover(ids.flip(1))
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        loss=masked_loss(replay.forward(ids,replay.capture(ids)),ids,[5,4]);loss.backward();optimizer.step()
        self.assertEqual(replay.frozen_hash(),fingerprint)
        self.assertTrue(check_masked_replay(replay,ids,[5,4])['all_masked_gradients_exact'])
        for name,p in model.named_parameters():
            if name in replay.suffix_names:p.requires_grad_(False)
        frozen_suffix=replay.frozen_suffix_hash()
        self.assertTrue(check_masked_replay(replay,ids,[5,4],True,5)['all_masked_gradients_exact'])
        frozen_optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        masked_loss(replay.forward(ids,replay.capture(ids)),ids,[5,4]).backward();frozen_optimizer.step()
        self.assertEqual(replay.frozen_suffix_hash(),frozen_suffix)
        self.assertEqual(replay.frozen_hash(),fingerprint)
        replay.close()

    def test_qwen3_cpu(self):self.check('qwen3','cpu')
    def test_qwen35_cpu(self):self.check('qwen35','cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
    def test_qwen3_cuda(self):self.check('qwen3','cuda')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA required')
    def test_qwen35_cuda(self):self.check('qwen35','cuda')
