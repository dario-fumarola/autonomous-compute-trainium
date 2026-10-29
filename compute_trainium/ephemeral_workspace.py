"""Untrained mechanism control: generated states used only at their own token.

This is a CPU reference, not a qualified accelerator recipe. It removes access
to earlier latent states from both source and latent queries. The authentic
source prefix, current generated state, shared parameters, rotary positions,
and natural next-token target remain. No state is detached.
"""
import math
import torch
from torch.nn import functional as F

from .interleaved_workspace import InterleavedWorkspace
from .probability_workspace import MathBlock


def ephemeral_allowed(length, device=None):
    if type(length) is not int or length < 1:
        raise ValueError('Positive integer source length required')
    past = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    zero = torch.zeros_like(past)
    own = torch.eye(length, dtype=torch.bool, device=device)
    # Storage [all source, all latent]. Sources never read a latent; latent t
    # reads source <= t and itself, but no other latent. This is not ordinary
    # causal attention in storage order.
    return torch.cat((torch.cat((past, zero), 1),
                      torch.cat((past, own), 1)), 0)


class EphemeralWorkspace(InterleavedWorkspace):
    def local_body(self, source, latent):
        if source.shape != latent.shape or source.ndim != 3:
            raise ValueError('Matching source/latent states required')
        n = source.shape[1]
        if not 0 < n <= self.config.max_seq:
            raise ValueError('Source length outside rotary cache')
        h = torch.cat((source, latent), 1)
        positions = torch.arange(n, device=h.device).repeat(2)
        cos = self.cos.index_select(2, positions).to(h.dtype)
        sin = self.sin.index_select(2, positions).to(h.dtype)
        allowed = ephemeral_allowed(n, h.device)
        for block in self.blocks:
            h = block(h, cos, sin, allowed)
        h = MathBlock.rms(h, self.norm) if self.explicit_attention else self.norm(h)
        return h[:, n:]

    def forward(self, tokens, *, return_states=False):
        source = self.embeddings(tokens)
        pos = torch.arange(source.shape[1], device=tokens.device)
        latent = self.body_at(source, pos)
        logits = F.linear(self.local_body(source, latent), self.prediction_weight)
        return (logits, latent) if return_states else logits

    def sequential_reference(self, tokens, *, return_states=False):
        """Recompute each source prefix plus only its current generated state."""
        source = self.embeddings(tokens)
        scores, states = [], []
        for t in range(source.shape[1]):
            prefix = source[:, :t+1]
            pos = torch.arange(t+1, device=tokens.device)
            latent = self.body_at(prefix, pos)[:, -1:]
            h = self.body_at(torch.cat((prefix, latent), 1),
                             torch.cat((pos, pos[-1:])))[:, -1:]
            scores.append(F.linear(h, self.prediction_weight))
            states.append(latent)
        result = torch.cat(scores, 1)
        return (result, torch.cat(states, 1)) if return_states else result

    def reused_source(self, tokens, *, return_states=False):
        """Same graph with ordinary source KVs reused, including their gradients.

        Two complete sets of projections, rather than recomputing the source
        body in the expanded reference. Dense latent attention is still charged
        at n queries by 2n keys despite the mask. CPU-only until Native qualified.
        """
        h = self.embeddings(tokens)
        b, n, d = h.shape
        cos, sin = self.cos[:, :, :n].to(h.dtype), self.sin[:, :, :n].to(h.dtype)
        past = torch.ones(n, n, dtype=torch.bool, device=h.device).tril()
        branch_allowed = torch.cat((past, torch.eye(n, dtype=torch.bool, device=h.device)), 1)

        def norm(x, layer):
            return MathBlock.rms(x, layer) if self.explicit_attention else layer(x)

        def rotate(a):
            even, odd = a[..., ::2], a[..., 1::2]
            return torch.stack((even*cos-odd*sin, odd*cos+even*sin), -1).flatten(-2)

        def project(x, block):
            z = norm(x, block.attn_norm)
            q = rotate(block.q(z).view(b, n, block.c.heads, -1).transpose(1, 2))
            k = rotate(block.k(z).view(b, n, block.c.kv_heads, -1).transpose(1, 2))
            v = block.v(z).view(b, n, block.c.kv_heads, -1).transpose(1, 2)
            return q, k, v

        def update(x, block, q, k, v, allowed):
            repeats = block.c.heads // block.c.kv_heads
            k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
            if self.explicit_attention:
                score = (q @ k.transpose(-1, -2)) / math.sqrt(d//block.c.heads)
                attended = score.masked_fill(~allowed, float('-inf')).softmax(-1) @ v
            else:
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            x = x + block.o(attended.transpose(1, 2).reshape(b, n, d))
            z = norm(x, block.ffn_norm)
            return x + block.down(F.silu(block.gate(z))*block.up(z))

        cache = []
        for block in self.blocks:
            q, k, v = project(h, block)
            cache.append((k, v))
            h = update(h, block, q, k, v, past)
        latent = norm(h, self.norm)
        h = latent
        for block, (source_k, source_v) in zip(self.blocks, cache):
            q, k, v = project(h, block)
            h = update(h, block, q, torch.cat((source_k, k), 2),
                       torch.cat((source_v, v), 2), branch_allowed)
        logits = F.linear(norm(h, self.norm), self.prediction_weight)
        return (logits, latent) if return_states else logits


def matrix_work(config, length, batch=1):
    """Actual full-square reference work, not a sparse/cached speed estimate."""
    if type(length) is not int or not 0 < length <= config.max_seq:
        raise ValueError('Positive bounded source length required')
    source = config.flops(0, length, batch)['forward_matmul_flops']
    expanded = config.flops(0, 2*length, batch)['forward_matmul_flops']
    head = 2*batch*length*config.dim*config.vocab_size
    forward = source + expanded + head
    return dict(forward_matmul_flops=forward, training_matmul_flops=3*forward,
                source_body_passes=1, expanded_body_passes=1,
                assumptions='Full-square masked attention, FMA=2, full backward; '
                            'no saving inferred from masked entries.')


def reused_source_matrix_work(config, length, batch=1):
    if type(length) is not int or not 0 < length <= config.max_seq:
        raise ValueError('Positive bounded source length required')
    # Source square plus latent n-by-2n attention. Source and latent each have
    # one set of projections/FFNs. Reused source K/V backward is accumulated.
    body = config.flops(0, length, batch)['forward_matmul_flops']
    extra_attention = batch * config.layers * 4 * length**2 * config.dim
    head = 2*batch*length*config.dim*config.vocab_size
    forward = 2*body + extra_attention + head
    return dict(forward_matmul_flops=forward, training_matmul_flops=3*forward,
                source_projection_passes=1, latent_projection_passes=1,
                assumptions='Dense n-by-n and n-by-2n attention; FMA=2, full '
                            'backward through reused source KV. No cache detach.')
