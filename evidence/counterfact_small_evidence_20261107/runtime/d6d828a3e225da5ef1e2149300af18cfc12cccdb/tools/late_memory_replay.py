"""Exact unpadded late-layer replay for Qwen3 and Qwen3.5 text backbones.

Discover the original suffix call arguments with hooks rather than reconstruct
architecture-specific rotary/mask/linear-attention rules. Frozen states, masks
and rotary templates are FP32; no KV state, dropout, padding or image inputs.
Validate logits and masked gradients against complete forwards before reuse.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

ENGRAM_HASH='b5f20018ee7ff73d0b43911830b0e8cd4c4540038c7c7775b821a6969b666630'


def load_engram(directory):
    path=Path(directory)/'engram.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ENGRAM_HASH:raise ValueError('pinned Engram source changed')
    name='_ngram_counterfact_engram';spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def hidden_arg(args,kwargs):
    return kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]


def replace_hidden(args,kwargs,h):
    if 'hidden_states' in kwargs:return args,{**kwargs,'hidden_states':h}
    return (h,*args[1:]),kwargs


class DenseMemoryAdapter(nn.Module):
    def __init__(self,hidden_size,budget):
        super().__init__();width=max(1,budget//(2*hidden_size))
        self.down=nn.Linear(hidden_size,width,bias=False);self.up=nn.Linear(width,hidden_size,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,hidden_states,input_ids):
        return self.up(torch.nn.functional.silu(self.down(hidden_states)))


class _Captured(Exception):
    def __init__(self,hidden):self.hidden=hidden


def detached(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad or value.is_inference():raise ValueError('suffix template retains gradients/inference tensor')
        return value.detach().clone()
    if isinstance(value,(tuple,list)):return type(value)(detached(x) for x in value)
    if isinstance(value,dict):return {k:detached(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise ValueError('stateful suffix argument cannot be cached: '+type(value).__name__)


def same_template(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,torch.Tensor):return a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_template(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same_template(x,y) for x,y in zip(a,b))
    return a==b


class LateMemoryReplay:
    def __init__(self,model,memory,layer_id):
        self.model=model;self.device=next(model.parameters()).device
        self.text=model.model.language_model if hasattr(model.model,'language_model') else model.model
        self.layers=self.text.layers;self.layer_id=layer_id;self.memory=memory
        if not 0<layer_id<len(self.layers):raise ValueError('nonempty prefix and suffix required')
        model.add_module('ngram_memory',memory)
        self.suffix_names=tuple(n for n,p in model.named_parameters() if any(n.startswith(f'{self.text_prefix()}.layers.{i}.') for i in range(layer_id,len(self.layers))))
        for n,p in model.named_parameters():
            if p.dtype!=torch.float32 or p.device!=self.device:raise ValueError('requires FP32 on one device')
            if p.requires_grad and not (n in self.suffix_names or n.startswith('ngram_memory.')):
                raise ValueError('prefix, head and vision parameters must be frozen')
        if any(isinstance(m,nn.Dropout) and m.p for m in model.modules()):raise ValueError('dropout prevents replay')
        self.ids=None;self.enabled=True;self.templates={}
        def inject(module,args,kwargs):
            if not self.enabled:return None
            if self.ids is None:raise RuntimeError('memory forward needs token context')
            h=hidden_arg(args,kwargs)
            delta=self.memory(hidden_states=h.unsqueeze(2),input_ids=self.ids.to(self.device)).squeeze(2)
            return replace_hidden(args,kwargs,h+delta)
        self.handle=self.layers[layer_id].register_forward_pre_hook(inject,with_kwargs=True)

    def text_prefix(self):return 'model.language_model' if hasattr(self.model.model,'language_model') else 'model'

    @contextmanager
    def context(self,ids,enabled=True):
        if ids.device.type!='cpu' or ids.dtype!=torch.int64 or ids.ndim!=2:raise ValueError('unpadded CPU int64 IDs required')
        prior=(self.ids,self.enabled);self.ids,self.enabled=ids,enabled
        try:yield
        finally:self.ids,self.enabled=prior

    def complete(self,ids,enabled=True,logits_to_keep=0):
        with self.context(ids,enabled):return self.model(input_ids=ids.to(self.device),use_cache=False,logits_to_keep=logits_to_keep).logits

    @torch.no_grad()
    def capture(self,ids):
        def stop(module,args,kwargs):raise _Captured(hidden_arg(args,kwargs).detach().clone())
        handle=self.layers[self.layer_id].register_forward_pre_hook(stop,with_kwargs=True,prepend=True)
        try:self.complete(ids,False,logits_to_keep=1)
        except _Captured as c:return c.hidden
        finally:handle.remove()
        raise RuntimeError('prefix was not intercepted')

    @torch.no_grad()
    def discover(self,ids):
        templates={};handles=[]
        for i in range(self.layer_id,len(self.layers)):
            def observe(module,args,kwargs,index=i):
                a,k=replace_hidden(args,kwargs,None);templates[index]=detached((a,k))
            handles.append(self.layers[i].register_forward_pre_hook(observe,with_kwargs=True,prepend=True))
        try:self.complete(ids,False,logits_to_keep=1)
        finally:
            for h in handles:h.remove()
        key=tuple(ids.shape)
        if key in self.templates and not same_template(self.templates[key],templates):
            raise AssertionError('suffix masks/positions depend on token content or mutable cache state')
        self.templates[key]=templates

    def forward(self,ids,hidden,enabled=True,logits_to_keep=0):
        if hidden.requires_grad or hidden.is_inference() or hidden.shape!=(*ids.shape,self.text.config.hidden_size) or hidden.dtype!=torch.float32 or hidden.device!=self.device:
            raise ValueError('matching ordinary detached prefixes required')
        key=tuple(ids.shape)
        if key not in self.templates:self.discover(ids)
        with self.context(ids,enabled):
            for i in range(self.layer_id,len(self.layers)):
                a,k=self.templates[key][i];a,k=replace_hidden(a,k,hidden)
                hidden=self.layers[i](*a,**k)
                if isinstance(hidden,tuple):hidden=hidden[0]
        normalized=self.text.norm(hidden)
        return self.model.lm_head(normalized[:,-logits_to_keep:] if logits_to_keep else normalized)

    def frozen_hash(self):
        h=hashlib.sha256()
        for n,p in self.model.state_dict().items():
            if n not in self.suffix_names and not n.startswith('ngram_memory.'):
                h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def frozen_suffix_hash(self):
        h=hashlib.sha256()
        for n,p in self.model.named_parameters():
            if n in self.suffix_names and not p.requires_grad:
                h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def close(self):self.handle.remove()


def masked_loss(logits,ids,starts):
    labels=ids.to(logits.device).clone()
    for i,start in enumerate(starts):labels[i,:start]=-100
    targets=labels[:,-(logits.shape[1]-1):]
    return torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),targets.reshape(-1))


def check_masked_replay(replay,ids,starts,enabled=True,logits_to_keep=0):
    model=replay.model;model.zero_grad(set_to_none=True)
    hidden=replay.capture(ids);replay.discover(ids)
    direct=replay.complete(ids,enabled,logits_to_keep);cached=replay.forward(ids,hidden,enabled,logits_to_keep)
    if not torch.equal(direct.detach().contiguous().view(torch.uint8),cached.detach().contiguous().view(torch.uint8)):
        raise AssertionError('replay changed complete logits')
    masked_loss(direct,ids,starts).backward()
    expected={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True);masked_loss(cached,ids,starts).backward()
    for n,p in model.named_parameters():
        if n in expected:
            a,b=expected[n],p.grad
            if (a is None)!=(b is None) or a is not None and not torch.equal(a,b):raise AssertionError('masked gradient changed: '+n)
        elif p.grad is not None:raise AssertionError('frozen parameter acquired gradient: '+n)
    model.zero_grad(set_to_none=True)
    return {'logits_byte_exact':True,'all_masked_gradients_exact':True,'trainable_gradients':len(expected)}
