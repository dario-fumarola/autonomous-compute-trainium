"""Causal probability-embedding replay, following PonderLM equations 7--8.

This is an independent implementation, not a reproduced published result.
Every proposal is generated without targets; the complete causal body is then
recomputed. No KV duplication, teacher token, or detached proposal is implicit.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import Block, LanguageModel


def expected_embedding(logits, weight):
    """Full vocabulary expectation; FP32 probabilities for low-precision input."""
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    return logits.to(dtype).softmax(-1) @ weight.to(dtype)


def replay(embedding, body, weight, extra_steps, proposal='expected'):
    if type(extra_steps) is not int or extra_steps < 0:
        raise ValueError('Nonnegative integer number of extra passes required')
    if proposal not in ('expected', 'uniform'):
        raise ValueError('Unknown proposal')
    state = embedding
    for _ in range(extra_steps):
        if proposal == 'expected':
            logits = F.linear(body(state), weight)
            thought = expected_embedding(logits, weight)
        else:
            # Explicit content-free control. It does NOT execute a proposal
            # body or vocabulary projection; its cheaper work must be counted.
            thought = weight.mean(0)
        state = state + thought.to(state.dtype)
    return F.linear(body(state), weight)


class MathBlock(Block):
    """Explicit attention and normalization for the Native reference path."""
    @staticmethod
    def rms(x, norm):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        y = x.to(dtype)
        return (y * (y.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
                * norm.weight.to(dtype)).to(x.dtype)

    def forward(self, x, cos, sin, attention_mask=None):
        b, t, d = x.shape
        z = self.rms(x, self.attn_norm)
        q = self.q(z).view(b, t, self.c.heads, -1).transpose(1, 2)
        k = self.k(z).view(b, t, self.c.kv_heads, -1).transpose(1, 2)
        v = self.v(z).view(b, t, self.c.kv_heads, -1).transpose(1, 2)
        def rotate(a):
            even, odd = a[..., ::2], a[..., 1::2]
            return torch.stack((even*cos-odd*sin, odd*cos+even*sin), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
        k = k.repeat_interleave(self.c.heads//self.c.kv_heads, 1)
        v = v.repeat_interleave(self.c.heads//self.c.kv_heads, 1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d//self.c.heads)
        if attention_mask is None:
            mask = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float('-inf'))
        elif attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float('-inf'))
        else:
            scores = scores + attention_mask
        h = scores.softmax(-1) @ v
        x = x + self.o(h.transpose(1, 2).reshape(b, t, d))
        z = self.rms(x, self.ffn_norm)
        return x + self.down(F.silu(self.gate(z))*self.up(z))


class ProbabilityWorkspace(LanguageModel):
    def __init__(self, config, *, explicit_attention=False):
        super().__init__(config)
        self.explicit_attention = explicit_attention
        if explicit_attention:
            # Preserve precisely the ordinary initializer, not a second draw.
            state = self.state_dict()
            self.blocks = nn.ModuleList([MathBlock(config) for _ in range(config.layers)])
            self.load_state_dict(state)

    def body(self, embedding):
        n = embedding.shape[1]
        if n > self.config.max_seq:
            raise ValueError('Input exceeds rotary cache')
        h = embedding
        cos, sin = self.cos[:, :, :n].to(h.dtype), self.sin[:, :, :n].to(h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin)
        return MathBlock.rms(h, self.norm) if self.explicit_attention else self.norm(h)

    def forward(self, tokens, extra_steps=0, proposal='expected'):
        return replay(self.embedding(tokens), self.body, self.embedding.weight,
                      extra_steps, proposal)


def matrix_work(config, length, extra_steps, proposal='expected', batch=1):
    """Executed dense matrix work, excluding elementwise/optimizer/communication.

    FMA=2; full-square attention; each differentiable matrix multiply's
    backward is twice forward; no activation checkpoint recomputation.
    Uniform proposal is a mean reduction, NOT a paid extra body pass.
    """
    if type(extra_steps) is not int or extra_steps < 0 or proposal not in ('expected', 'uniform'):
        raise ValueError('Invalid replay specification')
    base = config.flops(length, length, batch)
    k = extra_steps if proposal == 'expected' else 0
    embedding_projection = batch*2*length*config.dim*config.vocab_size*k
    forward = (k+1)*base['forward_matmul_flops']+embedding_projection
    return dict(forward_matmul_flops=forward, training_matmul_flops=3*forward,
                complete_body_passes=k+1, expected_embedding_forward_flops=embedding_projection,
                assumptions=base['assumptions'])
