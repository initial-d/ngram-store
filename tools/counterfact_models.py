"""Local Qwen3/Qwen3.5 text knowledge editing with pinned TinyEngram memory."""
import json
import hashlib
from pathlib import Path
import torch
from transformers import Qwen3ForCausalLM,Qwen3_5ForConditionalGeneration,AutoTokenizer
from .late_memory_replay import load_engram,LateMemoryReplay,DenseMemoryAdapter

class HeadConstantEmbedding(torch.nn.Module):
    """A separately trained vector per hash head, with no key-specific rows."""
    def __init__(self,source):
        super().__init__()
        self.num_heads=source.num_heads;self.embedding_dim=source.embedding_dim
        self.register_buffer('offsets',source.offsets.detach().clone())
        offsets=source.offsets.tolist()+[len(source.embedding.weight)]
        self.sizes=[b-a for a,b in zip(offsets,offsets[1:])]
        self.embedding=torch.nn.Embedding(self.num_heads,self.embedding_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.stack([source.embedding.weight[a:b].mean(0) for a,b in zip(offsets,offsets[1:])]))
    def forward(self,ids):
        return self.embedding(torch.arange(self.num_heads,device=ids.device).expand_as(ids))

class DisabledMemory(torch.nn.Module):
    def forward(self,**kwargs):
        raise RuntimeError('off control must explicitly disable memory')


def load_model(path):
    config=json.loads((Path(path)/'config.json').read_text())
    cls=Qwen3_5ForConditionalGeneration if config['model_type']=='qwen3_5' else Qwen3ForCausalLM
    model,info=cls.from_pretrained(path,local_files_only=True,dtype=torch.float32,
                                  attn_implementation='eager',output_loading_info=True)
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('backbone load failed: '+repr(info))
    if any(not (n.startswith('mtp.') or '.mtp.' in n) for n in info['unexpected_keys']):
        raise ValueError('unexpected backbone weights: '+repr(info['unexpected_keys']))
    return model,info


def setup(base,upstream,control,vocab_sizes,seed,device='cuda',suffix_layers=8):
    torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    model,info=load_model(base)
    text=model.model.language_model if hasattr(model.model,'language_model') else model.model
    layer=len(text.layers)-suffix_layers
    module=load_engram(upstream);module.set_global_step(0)
    cfg=module.EngramConfig(tokenizer_name_or_path=str(base),engram_vocab_size=vocab_sizes,
        max_ngram_size=4,n_embed_per_ngram=64,n_head_per_ngram=2,layer_ids=[layer],
        pad_id=tokenizer.pad_token_id,kernel_size=3,seed=seed)
    branch=module.Engram(layer_id=layer,config=cfg,backbone_hidden_size=text.config.hidden_size,backbone_hc_mult=1)
    # Match the embedding/linear initialization used by the pinned Qwen3
    # integration, while retaining Engram's deliberate zero convolution init.
    for name,child in branch.named_modules():
        if isinstance(child,(torch.nn.Linear,torch.nn.Embedding)):
            initial_seed=int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'little')
            generator=torch.Generator().manual_seed(initial_seed)
            torch.nn.init.normal_(child.weight,mean=0.,std=text.config.initializer_range,generator=generator)
            if isinstance(child,torch.nn.Linear) and child.bias is not None:torch.nn.init.zeros_(child.bias)
    budget=sum(p.numel() for p in branch.parameters())
    if control=='head-constant':branch.multi_head_embedding=HeadConstantEmbedding(branch.multi_head_embedding)
    elif control=='adapter':
        torch.manual_seed(int.from_bytes(hashlib.sha256(f'{seed}:adapter'.encode()).digest()[:8],'little'))
        branch=DenseMemoryAdapter(text.config.hidden_size,budget)
    elif control=='off':branch=DisabledMemory()
    prefix='model.language_model' if hasattr(model.model,'language_model') else 'model'
    suffix=tuple(f'{prefix}.layers.{i}.' for i in range(layer,len(text.layers)))
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(suffix))
    for p in branch.parameters():p.requires_grad_(control!='off')
    model.to(device).eval();branch.to(device)
    replay=LateMemoryReplay(model,branch,layer)
    return model,tokenizer,replay,module,cfg,info
