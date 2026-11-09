"""Causal TinyEngram state alongside the decoder's native KV/recurrent cache."""
from contextlib import contextmanager
import math

import torch


class CachedMemoryDecode(torch.nn.Module):
    """Retain only gated convolution history and the token n-gram suffix.

    This is the pinned TinyEngram inference formula, applied to new hidden
    positions. It uses the unchanged model's projection/gate/convolution
    modules; source and codec readers execute the same path. Padding, packed
    batches, beam reorder and speculative rollback require separate support.
    """
    def __init__(self, memory, *, global_step):
        super().__init__()
        self.branch = memory.source if hasattr(memory, 'source') else memory
        if not hasattr(self.branch, 'short_conv'):
            raise ValueError('pinned TinyEngram branch required')
        self.global_step = global_step
        self.value_history = None
        self.tokens_seen = 0
        self.batch_size = None
        convolution = self.branch.short_conv.conv
        if (convolution.stride != (1,) or convolution.bias is not None
                or convolution.padding != ((convolution.kernel_size[0] - 1) * convolution.dilation[0],)):
            raise ValueError('unsupported causal memory convolution')
        self.history_length = (convolution.kernel_size[0] - 1) * convolution.dilation[0]

    def reset(self):
        self.value_history = None; self.tokens_seen = 0; self.batch_size = None

    @torch.no_grad()
    def forward(self, hidden_states, input_ids):
        batch, new_tokens, groups, width = hidden_states.shape
        if input_ids.ndim != 2 or input_ids.shape != (batch, self.tokens_seen + new_tokens) or new_tokens < 1:
            raise ValueError('append-only full token context required')
        if self.batch_size is not None and self.batch_size != batch:
            raise ValueError('cached batch size changed without reset')
        branch = self.branch; device = hidden_states.device
        context_length = branch.config.max_ngram_size - 1 + new_tokens
        context = input_ids[:, -context_length:].detach().cpu().numpy()
        local_ids = branch.hash_mapping.hash(context)[branch.layer_id][:, -new_tokens:]
        embeddings = branch.multi_head_embedding(torch.from_numpy(local_ids).to(device)).flatten(start_dim=-2)
        gates = []
        for head in range(branch.backbone_hc_mult):
            key = branch.norm1[head](branch.key_projs[head](embeddings))
            query = branch.norm2[head](hidden_states[:, :, head, :])
            gate = (key * query).sum(dim=-1) / math.sqrt(branch.backbone_hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid().unsqueeze(-1))
        gates = torch.stack(gates, dim=2)
        if self.global_step < branch.config.warmup_steps:
            gates = torch.ones_like(gates)
        elif self.global_step < branch.config.warmup_steps + branch.config.soft_constraint_steps:
            gates = gates.clamp_min(.1)
        value = gates * branch.value_proj(embeddings).unsqueeze(2)
        context_value = value if self.value_history is None else torch.cat((self.value_history, value), dim=1)
        output = value + branch.short_conv(context_value)[:, -new_tokens:]
        self.value_history = context_value[:, -self.history_length:].clone() if self.history_length else None
        self.tokens_seen += new_tokens; self.batch_size = batch
        return output


@contextmanager
def cached_memory_session(replay, global_step):
    original = replay.memory
    cached = CachedMemoryDecode(original, global_step=global_step)
    replay.memory = cached; replay.model.ngram_memory = cached
    try: yield cached
    finally:
        replay.memory = original; replay.model.ngram_memory = original; cached.reset()


@torch.no_grad()
def generate_cached(replay, prefix, *, steps, global_step, continuation=None):
    """Real decoder cache, greedy generation or fixed matched continuation."""
    if steps < 1 or prefix.dtype != torch.int64 or prefix.device.type != 'cpu' or prefix.ndim != 2:
        raise ValueError('nonempty CPU prefix and positive decode steps required')
    if continuation is not None and continuation.shape != (prefix.shape[0], steps):
        raise ValueError('matched continuation shape differs')
    full_ids = prefix; outputs = []; generated = []
    with cached_memory_session(replay, global_step):
        with replay.context(full_ids):
            result = replay.model(input_ids=prefix.to(replay.device), use_cache=True, logits_to_keep=1)
        outputs.append(result.logits.detach().cpu())
        for step in range(steps):
            token = result.logits[:, -1].argmax(-1).cpu() if continuation is None else continuation[:, step]
            generated.append(token)
            full_ids = torch.cat((full_ids, token[:, None]), dim=1)
            with replay.context(full_ids):
                result = replay.model(input_ids=token[:, None].to(replay.device),
                    past_key_values=result.past_key_values, use_cache=True, logits_to_keep=1)
            outputs.append(result.logits.detach().cpu())
    return torch.stack(generated, dim=1), outputs
