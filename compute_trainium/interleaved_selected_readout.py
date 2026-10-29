"""Exact CPU latent history with vocabulary projections only at requested exits."""
import torch
from torch.nn import functional as F


def cached_selected_logits(model, tokens, prediction_positions):
    positions = list(prediction_positions)
    n = tokens.shape[1]
    if (model.explicit_attention or n < 1 or not positions
            or any(type(p) is not int or not 0 <= p < n for p in positions)
            or positions != sorted(set(positions))):
        raise ValueError('Nonempty CPU prefix and increasing distinct valid positions required')
    source = model.embeddings(tokens)
    caches = [None]*len(model.blocks)
    selected = set(positions)
    outputs = []

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
            repeat = block.c.heads//block.c.kv_heads
            attended = F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, 1),
                                                       v.repeat_interleave(repeat, 1), is_causal=False)
            h = h + block.o(attended.transpose(1, 2).reshape(b, 1, d))
            normalized = block.ffn_norm(h)
            h = h + block.down(F.silu(block.gate(normalized))*block.up(normalized))
        return model.norm(h)

    for t in range(n):
        latent = step(source[:, t:t+1], t)
        h = step(latent, t)
        if t in selected:
            outputs.append(h)
    return F.linear(torch.cat(outputs, 1), model.prediction_weight)
