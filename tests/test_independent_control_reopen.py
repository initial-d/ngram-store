"""Completed independent controls need trained deltas, not the original table."""
import json
import unittest

from tests import test_counterfact_native as fixture


@unittest.skipUnless(not getattr(fixture.CounterfactNativeTest, '__unittest_skip__', False), 'Qwen fixture required')
class IndependentReopenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CounterfactNativeTest.setUpClass(); cls.helper=fixture.CounterfactNativeTest()

    @classmethod
    def tearDownClass(cls): fixture.CounterfactNativeTest.tearDownClass()

    def test_trained_constants_and_adapter_reopen_after_vocabulary_removal(self):
        from safetensors.torch import save_file
        from tools.counterfact_models import HeadConstantEmbedding, ConstantAddressMapping
        from tools.counterfact_native import restore_independent_control, digest
        from tools.cached_memory_decode import generate_cached
        from tools.late_memory_replay import DenseMemoryAdapter
        from tools.tree_node_quality import write, digest as file_digest
        torch=fixture.torch
        for control in ('head-constant','adapter'):
            with self.subTest(control=control):
                run,model,replay=self.helper.fixture('exact')
                try:
                    budget=sum(p.numel() for p in replay.memory.parameters())
                    if control=='head-constant':
                        replay.memory.multi_head_embedding=HeadConstantEmbedding(replay.memory.multi_head_embedding)
                        replay.memory.hash_mapping=ConstantAddressMapping(replay.layer_id,6)
                        identity={'row_bytes':6*32*4}
                    else:
                        adapter=DenseMemoryAdapter(replay.text.config.hidden_size,budget)
                        replay.memory=adapter;model.ngram_memory=adapter
                        identity={'target_parameter_budget':budget,'row_bytes':0}
                    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
                    ids=torch.tensor([[3,4,5,6,7]])
                    for _ in range(3):
                        optimizer.zero_grad();replay.complete(ids,True,1).square().mean().backward();optimizer.step()
                    state={n:p.detach().contiguous() for n,p in model.named_parameters() if p.requires_grad}
                    save_file(state,str(run/'delta.safetensors'))
                    protocol={'independent_identity_controls':True,'suffix_layers':8,'train_decoder_suffix':False,
                        'variants':{control:{'control':control,'vocab_sizes':[257,521,1031]}}}
                    write(run/'report.json',{'status':'complete','protocol':protocol,'variant':control,'seed':3,
                        'base_model':str(self.helper.base),'optimizer_steps':3,'identity':identity,
                        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'frozen_weight_sha256':replay.frozen_hash(),'frozen_decoder_suffix_sha256':replay.frozen_suffix_hash(),
                        'delta':{'sha256':file_digest(run/'delta.safetensors'),'bytes':(run/'delta.safetensors').stat().st_size}})
                    (run/'vocabulary.npz').unlink()
                    _,other_model,other=restore_independent_control(run,fixture.SOURCE,device='cpu')
                    try:
                        with torch.no_grad():
                            self.assertEqual(digest(replay.complete(ids,True,1)),digest(other.complete(ids,True,1)))
                            reference,values=generate_cached(replay,ids,steps=4,global_step=3)
                            generated,actual=generate_cached(other,ids,steps=4,global_step=3)
                            self.assertEqual(digest(reference),digest(generated))
                            self.assertEqual([digest(x) for x in values],[digest(x) for x in actual])
                    finally:other.close()
                finally:replay.close()


if __name__=='__main__':unittest.main()
