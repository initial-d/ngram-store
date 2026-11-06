import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    from tools.ngram_identity import Vocabulary,IdentityMapping,collision_audit,configure_memory
except ImportError:
    np=None
try:
    import torch
except ImportError:
    torch=None


class TokenMap:
    def __init__(self):self.lookup_table=np.arange(128,dtype=np.int64)
    def __call__(self,x):return self.lookup_table[np.asarray(x,dtype=np.int64)]


class Mapping:
    def __init__(self):
        self.pad_id=0;self.layer_ids=[2];self.compressed_tokenizer=TokenMap()
        self.vocab_size_per_ngram=[3,3,3];self.vocab_size_across_layers=self.calculate_vocab_size_across_layers()
    def calculate_vocab_size_across_layers(self):
        return {2:[[max(2,u),max(2,u)+1] for u in self.vocab_size_per_ngram]}
    def _get_ngram_hashes(self,x,layer):
        # Deliberately colliding fixture, not an empirical Engram benchmark.
        output=[]
        for heads in self.vocab_size_across_layers[layer]:
            for size in heads:output.append(x%2%size)
        return np.stack(output,axis=-1)


@unittest.skipUnless(np is not None,'NumPy required')
class IdentityTest(unittest.TestCase):
    def fixture(self):
        source=Mapping();payload={'train':[0], 'sequences':[{'ids':[1,2,3,99]}, {'ids':[7,8,9,99]}]}
        return source,Vocabulary.build(payload,source,'prepared')
    def test_training_only_causal_keys(self):
        source,v=self.fixture()
        self.assertEqual(v.keys[2].tolist(),[[0,1],[1,2],[2,3]])
        self.assertNotIn((3,99),v.indices[2])
        self.assertNotIn((7,8),v.indices[2])
        self.assertTrue(np.all(v.known_ids(np.array([[1,2,3]]))!=0))
        self.assertTrue(np.all(v.known_ids(np.array([[7,8,9]]))==0))
    def test_oov_rules_and_preserved_normalization(self):
        source,v=self.fixture();h=IdentityMapping(source,v,False);e=IdentityMapping(source,v,True)
        x=np.array([[7,8]])
        np.testing.assert_array_equal(h.hash(x)[2],e.hash(x)[2])
        h.native_oov=True;self.assertTrue(np.all(h.hash(x)[2]>0))
        source.compressed_tokenizer.lookup_table[3]=2
        with self.assertRaisesRegex(ValueError,'normalization'):IdentityMapping(source,v,True)
    def test_saved_identity_and_individual_vs_joint_collisions(self):
        source,v=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'vocabulary.npz';v.save(p);r=Vocabulary.load(p)
            self.assertEqual(v.indices,r.indices)
            np.testing.assert_array_equal(v.counts[3],r.counts[3])
        audit=collision_audit(v,source)
        self.assertEqual(audit['2']['unique_training_keys'],3)
        self.assertEqual(audit['2']['distinct_joint_head_addresses'],2)
        self.assertEqual(audit['2']['heads'][0]['keys_in_shared_buckets'],2)


@unittest.skipUnless(np is not None and torch is not None,'NumPy and Torch required')
class PairedEmbeddingTest(unittest.TestCase):
    def test_initial_vectors_gradients_and_frozen_unknown(self):
        class Multi(torch.nn.Module):
            def __init__(self,sizes,d):
                super().__init__();self.embedding=torch.nn.Embedding(sum(sizes),d)
                self.register_buffer('offsets',torch.tensor([sum(sizes[:i]) for i in range(len(sizes))]))
            def forward(self,ids):return self.embedding(ids+self.offsets)
        module=SimpleNamespace(MultiHeadEmbedding=Multi)
        payload={'train':[0],'sequences':[{'ids':[1,2,3,99]}]};source=Mapping()
        v=Vocabulary.build(payload,source,'prepared');objects=[]
        for exact in (False,True):
            memory=SimpleNamespace(hash_mapping=Mapping(),layer_id=2)
            replay=SimpleNamespace(memory=memory,device=torch.device('cpu'),text=SimpleNamespace(config=SimpleNamespace(initializer_range=.02)))
            configure_memory(replay,module,v,exact,3);objects.append(memory)
        x=np.array([[1,2,3,7]])
        hm,em=objects;hids=torch.from_numpy(hm.hash_mapping.hash(x)[2]);eids=torch.from_numpy(em.hash_mapping.hash(x)[2])
        a=hm.multi_head_embedding(hids);b=em.multi_head_embedding(eids)
        self.assertTrue(torch.equal(a,b))
        a.sum().backward();b.sum().backward()
        target=hm.multi_head_embedding.embedding.weight.grad
        pooled=torch.zeros_like(target)
        for head,(hoff,eoff) in enumerate(zip(hm.multi_head_embedding.offsets.tolist(),em.multi_head_embedding.offsets.tolist())):
            n=2+head//2;local=hm.hash_mapping.source._get_ngram_hashes(v.keys[n],2)[:,-1,head]+1
            pooled.index_add_(0,torch.from_numpy(local+hoff),em.multi_head_embedding.embedding.weight.grad[eoff+1:eoff+1+len(local)])
        self.assertTrue(torch.equal(target,pooled))
        for memory in objects:
            embedding=memory.multi_head_embedding
            torch.optim.Adam(embedding.parameters(),lr=.1).step()
            self.assertTrue(torch.equal(embedding.embedding.weight[embedding.offsets],torch.zeros(6,32)))


if __name__=='__main__':unittest.main()
