"""FP32 CPU/CUDA training reuse for a late TinyEngram branch on frozen Qwen.

Capture the state immediately before the branch in the original pinned forward.
Resume the SAME decoder suffix and full-vocabulary head. This saves frozen
prefix work across epochs, not inference memory or native query costs. No KV
cache, padding, dropout, altered objective or quantized hidden states.
"""
import hashlib

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from .device_row_adapter import StagedMultiHeadEmbedding


class _CapturedPrefix(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


class FrozenTinyEngramPrefix:
    def __init__(self, model):
        self.model = model
        self.backbone = model.model
        self.device = next(model.parameters()).device
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("prefix reuse supports CPU and CUDA")
        layers = list(self.backbone.engram_layers)
        if len(layers) != 1:
            raise ValueError("requires exactly one memory branch")
        self.layer_id = int(layers[0])
        if not 0 < self.layer_id < model.config.num_hidden_layers:
            raise ValueError("requires a nonempty frozen prefix and decoder suffix")
        self.branch = self.backbone.engram_layers[layers[0]]
        if model.config._attn_implementation != "eager":
            raise ValueError("requires pinned eager attention")
        for name, p in model.named_parameters():
            if p.device != self.device or p.dtype != torch.float32:
                raise ValueError("requires FP32 parameters on one fixed device")
            if "engram_layers" not in name and p.requires_grad:
                raise ValueError("all non-memory parameters must be frozen")
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
            raise ValueError("dropout is incompatible with reusable frozen states")
        if model.config.attention_dropout:
            raise ValueError("attention dropout is incompatible with reusable frozen states")

    def frozen_weight_sha256(self):
        digest = hashlib.sha256()
        for name, p in self.model.state_dict().items():
            if "engram_layers" not in name:
                digest.update(name.encode())
                digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def capture(self, ids):
        self._validate(ids)

        def stop(module, args, kwargs):
            raise _CapturedPrefix(kwargs["hidden_states"].squeeze(2).detach().clone())

        handle = self.branch.register_forward_pre_hook(stop, with_kwargs=True)
        try:
            self.backbone(input_ids=ids.to(self.device), use_cache=False)
        except _CapturedPrefix as captured:
            return captured.hidden
        finally:
            handle.remove()
        raise RuntimeError("memory branch was skipped; enable upstream Engram before capture")

    def _validate(self, ids):
        if (ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != 2
                or not all(ids.shape)):
            raise ValueError("requires nonempty unpadded CPU int64 token IDs")

    def forward(self, ids, hidden, *, use_memory=True):
        self._validate(ids)
        if (hidden.shape != (*ids.shape, self.model.config.hidden_size)
                or hidden.device != self.device or hidden.dtype != torch.float32
                or hidden.requires_grad or hidden.is_inference()):
            raise ValueError("requires ordinary detached FP32 prefix states with matching shape")
        cache_position = torch.arange(ids.shape[1],device=self.device)
        position_ids = cache_position.unsqueeze(0)
        mask_kwargs = dict(config=self.model.config, input_embeds=hidden,
                           attention_mask=None, cache_position=cache_position,
                           past_key_values=None, position_ids=position_ids)
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.backbone.has_sliding_layers:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        rotations = self.backbone.rotary_emb(hidden, position_ids)
        if use_memory:
            # Native staging accepts CPU hash IDs. Let the pinned Engram layer
            # hash CPU tokens for that adapter; its returned vectors are CUDA.
            # Ordinary trainable CUDA embeddings keep upstream CUDA-ID behavior.
            tokens = ids if isinstance(self.branch.multi_head_embedding,StagedMultiHeadEmbedding) else ids.to(self.device)
            hidden = hidden + self.branch(hidden_states=hidden.unsqueeze(2), input_ids=tokens).squeeze(2)
        for layer in self.backbone.layers[self.layer_id:]:
            hidden = layer(hidden, attention_mask=masks[layer.attention_type],
                           position_ids=position_ids, past_key_values=None, use_cache=False,
                           cache_position=cache_position, position_embeddings=rotations)
        return self.model.lm_head(self.backbone.norm(hidden))


def check_reuse(model, reuse, ids, *, gradients=False):
    """Compare full logits and, optionally, every branch gradient before reuse."""
    model.zero_grad(set_to_none=True)
    hidden = reuse.capture(ids)
    direct = model(ids.to(reuse.device), use_cache=False).logits
    cached = reuse.forward(ids, hidden)
    if not torch.equal(direct.detach().contiguous().view(torch.uint8),
                       cached.detach().contiguous().view(torch.uint8)):
        raise AssertionError("reused prefix changed matched-schedule logits")
    result = {"logits_byte_exact": True, "targets": ids.shape[0]*(ids.shape[1]-1)}
    if gradients:
        def loss(logits):
            return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                     ids[:, 1:].to(reuse.device).reshape(-1))
        loss(direct).backward()
        expected = {n: p.grad.clone() if p.grad is not None else None
                    for n, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        loss(cached).backward()
        for name, p in model.named_parameters():
            if name in expected:
                a, b = expected[name], p.grad
                if ((a is None) != (b is None) or a is not None and not torch.equal(a, b)):
                    raise AssertionError("reused prefix changed gradient: " + name)
            elif p.grad is not None:
                raise AssertionError("frozen backbone acquired gradients")
        model.zero_grad(set_to_none=True)
        result["branch_gradients_exact"] = True
    return result
