"""Saved-weight, CPU-only causal-path interventions; not new trained models.

Every condition computes a source step and a latent step at each position.
Changing a latent input always recomputes its K/V and all dependent future states.
The current-only condition masks earlier latent slots for BOTH source and latent
queries, preventing hidden persistence through source K/V. Corruption can create
distribution shift; a loss increase alone does not establish reasoning.
"""
import torch
from torch.nn import functional as F

MODES = ('authentic', 'current_only', 'lagged', 'constant')


def _validate(model, tokens, mode, positions):
    if model.explicit_attention or tokens.device.type != 'cpu':
        raise ValueError('CPU SDPA model required')
    if mode not in MODES:
        raise ValueError('Unknown content intervention')
    if tokens.ndim != 2 or not 0 < tokens.shape[1] <= model.config.max_seq:
        raise ValueError('Nonempty bounded source sequence required')
    if (not positions or positions != sorted(set(positions)) or
            any(type(p) is not int or not 0 <= p < tokens.shape[1] for p in positions)):
        raise ValueError('Increasing distinct valid readout positions required')


def _allowed(length, mode, *, last_only, device):
    keys = torch.arange(length, device=device)
    queries = keys[-1:] if last_only else keys
    mask = keys[None, :] <= queries[:, None]
    if mode == 'current_only':
        mask = mask & ((keys[None, :] % 2 == 0) | (keys[None, :] == queries[:, None]))
    return mask


def _input(model, generated, previous_generated, mode, constant=None):
    if mode == 'constant':
        return constant.expand_as(generated)
    if mode == 'lagged' and previous_generated is not None:
        return previous_generated
    return generated


def cached_intervention(model, tokens, *, mode='authentic', positions=None,
                        return_states=False):
    positions = list(range(tokens.shape[1])) if positions is None else list(positions)
    _validate(model, tokens, mode, positions)
    source = model.embeddings(tokens)
    constant = model.source_scale * model.embedding.weight.mean(0) if mode == 'constant' else None
    caches = [None] * len(model.blocks)
    selected, outputs, states, used_states = set(positions), [], [], []
    previous_generated = None

    def step(h, position):
        batch, _, dim = h.shape
        cos = model.cos[:, :, position:position + 1].to(h.dtype)
        sin = model.sin[:, :, position:position + 1].to(h.dtype)

        def rotate(x):
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2)

        for i, block in enumerate(model.blocks):
            n = block.attn_norm(h)
            q = rotate(block.q(n).view(batch, 1, block.c.heads, -1).transpose(1, 2))
            k = rotate(block.k(n).view(batch, 1, block.c.kv_heads, -1).transpose(1, 2))
            v = block.v(n).view(batch, 1, block.c.kv_heads, -1).transpose(1, 2)
            if caches[i] is not None:
                k, v = torch.cat((caches[i][0], k), 2), torch.cat((caches[i][1], v), 2)
            caches[i] = (k, v)
            repeat = block.c.heads // block.c.kv_heads
            allowed = _allowed(k.shape[2], mode, last_only=True, device=h.device)
            a = F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, 1),
                                               v.repeat_interleave(repeat, 1), attn_mask=allowed)
            h = h + block.o(a.transpose(1, 2).reshape(batch, 1, dim))
            n = block.ffn_norm(h)
            h = h + block.down(F.silu(block.gate(n)) * block.up(n))
        return model.norm(h)

    for t in range(tokens.shape[1]):
        generated = step(source[:, t:t + 1], t)
        used = _input(model, generated, previous_generated, mode, constant)
        result = step(used, t)
        previous_generated = generated
        states.append(generated)
        used_states.append(used)
        if t in selected:
            outputs.append(result)
    logits = F.linear(torch.cat(outputs, 1), model.prediction_weight)
    if return_states:
        return logits, torch.cat(states, 1), torch.cat(used_states, 1)
    return logits


def recomputed_intervention(model, tokens, *, mode='authentic', positions=None,
                           return_states=False):
    """Independent full-prefix attention reference, including all replacements."""
    positions = list(range(tokens.shape[1])) if positions is None else list(positions)
    _validate(model, tokens, mode, positions)
    source = model.embeddings(tokens)
    constant = model.source_scale * model.embedding.weight.mean(0) if mode == 'constant' else None
    history, rotary, states, used_states, outputs = [], [], [], [], []
    previous_generated = None
    selected = set(positions)

    def body():
        h = torch.cat(history, 1)
        pos = torch.tensor(rotary, device=h.device)
        cos = model.cos.index_select(2, pos).to(h.dtype)
        sin = model.sin.index_select(2, pos).to(h.dtype)
        allowed = _allowed(h.shape[1], mode, last_only=False, device=h.device)
        for block in model.blocks:
            h = block(h, cos, sin, allowed)
        return model.norm(h)[:, -1:]

    for t in range(tokens.shape[1]):
        history.append(source[:, t:t + 1]); rotary.append(t)
        generated = body()
        used = _input(model, generated, previous_generated, mode, constant)
        history.append(used); rotary.append(t)
        result = body()
        previous_generated = generated
        states.append(generated); used_states.append(used)
        if t in selected:
            outputs.append(result)
    logits = F.linear(torch.cat(outputs, 1), model.prediction_weight)
    if return_states:
        return logits, torch.cat(states, 1), torch.cat(used_states, 1)
    return logits
