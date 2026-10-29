"""CPU cached sequential reference; no accelerator-serving qualification.

One generated latent follows each authentic source token. Each source and latent
has the same rotary index, but its own causal KV entry. Cached tensors stay on
the autograd graph so tests can compare complete gradients with recomputation.
"""
import torch
from torch.nn import functional as F


def cached_sequential(model, tokens, *, return_states=False):
    if model.explicit_attention:
        raise ValueError('Independent CPU SDPA model required')
    source = model.embeddings(tokens)
    caches = [None] * len(model.blocks)
    states, scores = [], []

    def step(h, position):
        b, _, d = h.shape
        cos = model.cos[:, :, position:position+1].to(h.dtype)
        sin = model.sin[:, :, position:position+1].to(h.dtype)
        def rotate(a):
            even, odd = a[..., ::2], a[..., 1::2]
            return torch.stack((even*cos-odd*sin, odd*cos+even*sin), -1).flatten(-2)
        for i, block in enumerate(model.blocks):
            normalized = block.attn_norm(h)
            q = rotate(block.q(normalized).view(b, 1, block.c.heads, -1).transpose(1, 2))
            k = rotate(block.k(normalized).view(b, 1, block.c.kv_heads, -1).transpose(1, 2))
            v = block.v(normalized).view(b, 1, block.c.kv_heads, -1).transpose(1, 2)
            if caches[i] is not None:
                k = torch.cat((caches[i][0], k), 2)
                v = torch.cat((caches[i][1], v), 2)
            caches[i] = (k, v)
            repeat = block.c.heads // block.c.kv_heads
            attended = F.scaled_dot_product_attention(
                q, k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1),
                is_causal=False)  # Every cached entry is in this query's past.
            h = h + block.o(attended.transpose(1, 2).reshape(b, 1, d))
            normalized = block.ffn_norm(h)
            h = h + block.down(F.silu(block.gate(normalized))*block.up(normalized))
        return model.norm(h)

    for t in range(source.shape[1]):
        latent = step(source[:, t:t+1], t)
        states.append(latent)
        scores.append(F.linear(step(latent, t), model.prediction_weight))
    logits, states = torch.cat(scores, 1), torch.cat(states, 1)
    return (logits, states) if return_states else logits
