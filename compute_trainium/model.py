"""A small Llama-style reference with explicit real-target projection."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 768
    layers: int = 20
    heads: int = 12
    kv_heads: int = 4
    ffn_dim: int = 2048
    vocab_size: int = 32000
    max_seq: int = 4096

    def __post_init__(self):
        if (min(self.dim, self.layers, self.heads, self.kv_heads, self.ffn_dim,
                self.vocab_size, self.max_seq) < 1 or self.dim % self.heads or
                self.heads % self.kv_heads or (self.dim // self.heads) % 2):
            raise ValueError("Invalid grouped-query model dimensions.")

    def parameter_count(self):
        kv = self.kv_heads * (self.dim // self.heads)
        return (self.vocab_size * self.dim + self.dim + self.layers *
                (2 * self.dim ** 2 + 2 * self.dim * kv +
                 3 * self.dim * self.ffn_dim + 2 * self.dim))

    def flops(self, real_targets: int, expanded_positions: int, batch: int = 1):
        """Dense matmul estimate; multiply/add=2, backward=2x forward, no AC."""
        d, f, n = self.dim, self.ffn_dim, expanded_positions
        kv = self.kv_heads * (d // self.heads)
        projections = self.layers * n * (4*d*d + 4*d*kv + 6*d*f)
        # Conservative full-square attention; no assumed triangular kernel saving.
        attention = self.layers * 4*n*n*d
        head = 2 * real_targets * d * self.vocab_size
        forward = batch * (projections + attention + head)
        return {"forward_matmul_flops": forward, "training_matmul_flops": 3 * forward,
                "lm_head_forward_flops": batch * head,
                "assumptions": "Full-square attention, FMA=2, backward=2x forward; excludes elementwise/optimizer/AC. Not a hardware counter."}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.attn_norm = nn.RMSNorm(c.dim, eps=1e-5)
        self.ffn_norm = nn.RMSNorm(c.dim, eps=1e-5)
        kv = c.kv_heads * (c.dim // c.heads)
        self.q = nn.Linear(c.dim, c.dim, bias=False)
        self.k = nn.Linear(c.dim, kv, bias=False)
        self.v = nn.Linear(c.dim, kv, bias=False)
        self.o = nn.Linear(c.dim, c.dim, bias=False)
        self.gate = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.up = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.down = nn.Linear(c.ffn_dim, c.dim, bias=False)

    def forward(self, x, cos, sin, attention_mask=None):
        b, t, d = x.shape
        z = self.attn_norm(x)
        q = self.q(z).view(b, t, self.c.heads, -1).transpose(1, 2)
        k = self.k(z).view(b, t, self.c.kv_heads, -1).transpose(1, 2)
        v = self.v(z).view(b, t, self.c.kv_heads, -1).transpose(1, 2)
        def rotate(a):
            even, odd = a[..., ::2], a[..., 1::2]
            return torch.stack((even*cos - odd*sin, odd*cos + even*sin), dim=-1).flatten(-2)
        q, k = rotate(q), rotate(k)
        k = k.repeat_interleave(self.c.heads // self.c.kv_heads, dim=1)
        v = v.repeat_interleave(self.c.heads // self.c.kv_heads, dim=1)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                           is_causal=attention_mask is None)
        x = x + self.o(h.transpose(1, 2).reshape(b, t, d))
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.RMSNorm(config.dim, eps=1e-5)
        frequency = 10000 ** (-torch.arange(0, config.dim // config.heads, 2).float()
                              / (config.dim // config.heads))
        phase = torch.arange(config.max_seq).float()[:, None] * frequency[None, :]
        self.register_buffer("cos", phase.cos()[None, None], persistent=False)
        self.register_buffer("sin", phase.sin()[None, None], persistent=False)
        for name, p in self.named_parameters():
            if p.ndim == 1:
                nn.init.ones_(p)
            else:
                std = .02 / math.sqrt(2*config.layers) if name.endswith(("o.weight", "down.weight")) else .02
                nn.init.normal_(p, std=std)

    def hidden(self, tokens, attention_mask=None):
        n = tokens.shape[1]
        if n > self.config.max_seq:
            raise ValueError("Input exceeds the static rotary cache.")
        h = self.embedding(tokens)
        cos, sin = self.cos[:, :, :n].to(h.dtype), self.sin[:, :, :n].to(h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin, attention_mask)
        return self.norm(h)

    def forward(self, tokens, prediction_positions=None, attention_mask=None):
        h = self.hidden(tokens, attention_mask)
        if prediction_positions is not None:
            h = h.index_select(1, prediction_positions)
        return F.linear(h, self.embedding.weight)
