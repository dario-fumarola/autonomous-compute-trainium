"""Exact CPU cached readout for explicitly supported latent-token budgets.

No accelerator serving or trained budget support is asserted. Caches retain
their full gradient graph so software tests can compare against recomputation.
"""
import torch
from torch.nn import functional as F

from .budgeted_interleaved_reference import check_budget


def cached_budget(model, tokens, thought_budget, *, return_states=False):
    check_budget(thought_budget)
    if model.explicit_attention:
        raise ValueError('Independent CPU SDPA reference required')
    source = model.embeddings(tokens)
    caches = [None]*len(model.blocks)
    states = [[] for _ in range(thought_budget)]
    scores = []

    def step(h, position):
        b, _, d = h.shape
        cos = model.cos[:, :, position:position+1].to(h.dtype)
        sin = model.sin[:, :, position:position+1].to(h.dtype)
        def rotate(a):
            even, odd = a[..., ::2], a[..., 1::2]
            return torch.stack((even*cos-odd*sin, odd*cos+even*sin), -1).flatten(-2)
        for i, block in enumerate(model.blocks):
            z = block.attn_norm(h)
            q = rotate(block.q(z).view(b, 1, block.c.heads, -1).transpose(1, 2))
            k = rotate(block.k(z).view(b, 1, block.c.kv_heads, -1).transpose(1, 2))
            v = block.v(z).view(b, 1, block.c.kv_heads, -1).transpose(1, 2)
            if caches[i] is not None:
                k, v = torch.cat((caches[i][0], k), 2), torch.cat((caches[i][1], v), 2)
            caches[i] = k, v
            repeats = block.c.heads//block.c.kv_heads
            # The cache contains only this query and its chronological past.
            attended = F.scaled_dot_product_attention(q, k.repeat_interleave(repeats, 1),
                v.repeat_interleave(repeats, 1), is_causal=False)
            h = h+block.o(attended.transpose(1, 2).reshape(b, 1, d))
            z = block.ffn_norm(h)
            h = h+block.down(F.silu(block.gate(z))*block.up(z))
        return model.norm(h)

    for t in range(source.shape[1]):
        h = step(source[:, t:t+1], t)
        for slot in range(thought_budget):
            states[slot].append(h)
            h = step(h, t)
        scores.append(F.linear(h, model.prediction_weight))
    logits = torch.cat(scores, 1)
    return (logits, tuple(torch.cat(s, 1) for s in states)) if return_states else logits
