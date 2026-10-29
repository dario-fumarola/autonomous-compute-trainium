"""Exact depth-causal latent stages; no learned router or training admission.

Attention follows (source_position <= query_position, stage <= query_stage).
The full-mask implementation independently recomputes earlier stages. The
cached implementation reuses their differentiable keys and values exactly.
"""
import math

import torch
from torch.nn import functional as F

from .budgeted_interleaved_reference import check_budget
from .interleaved_workspace import InterleavedWorkspace
from .probability_workspace import MathBlock


def depth_mask(length, stages, device=None):
    if type(length) is not int or length < 1 or type(stages) is not int or not 1 <= stages <= 3:
        raise ValueError('Positive source length and one to three stages required')
    position = torch.arange(length, device=device).repeat(stages)
    stage = torch.arange(stages, device=device).repeat_interleave(length)
    return ((position[None, :] <= position[:, None]) &
            (stage[None, :] <= stage[:, None]))


class DepthCausalWorkspace(InterleavedWorkspace):
    def _full_body(self, inputs):
        n = inputs[0].shape[1]
        h = torch.cat(inputs, dim=1)
        positions = torch.arange(n, device=h.device).repeat(len(inputs))
        mask = depth_mask(n, len(inputs), h.device)
        cos, sin = self.cos.index_select(2, positions).to(h.dtype), self.sin.index_select(2, positions).to(h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin, mask)
        h = MathBlock.rms(h, self.norm) if self.explicit_attention else self.norm(h)
        return h.split(n, dim=1)

    def full_reference(self, tokens, budget=1, *, return_stages=False):
        check_budget(budget)
        inputs, outputs = [self.embeddings(tokens)], []
        for _ in range(budget+1):
            outputs.append(self._full_body(inputs)[-1])
            inputs.append(outputs[-1])
        logits = F.linear(outputs[-1], self.prediction_weight)
        return (logits, tuple(outputs)) if return_stages else logits

    def _cached_stage(self, h, cache, positions, *, detach_cache=False):
        b, n, d = h.shape
        cos = self.cos.index_select(2, positions).to(h.dtype)
        sin = self.sin.index_select(2, positions).to(h.dtype)

        def rotate(a):
            even, odd = a[..., ::2], a[..., 1::2]
            return torch.stack((even*cos-odd*sin, odd*cos+even*sin), -1).flatten(-2)

        for layer, block in enumerate(self.blocks):
            z = MathBlock.rms(h, block.attn_norm) if self.explicit_attention else block.attn_norm(h)
            q = rotate(block.q(z).view(b, n, block.c.heads, -1).transpose(1, 2))
            k = rotate(block.k(z).view(b, n, block.c.kv_heads, -1).transpose(1, 2))
            v = block.v(z).view(b, n, block.c.kv_heads, -1).transpose(1, 2)
            previous = cache[layer]
            keys = [x[0].detach() if detach_cache else x[0] for x in previous] + [k]
            values = [x[1].detach() if detach_cache else x[1] for x in previous] + [v]
            key_positions = torch.cat([x[2] for x in previous]+[positions])
            allowed = key_positions[None, :] <= positions[:, None]
            repeats = block.c.heads//block.c.kv_heads
            all_k = torch.cat(keys, 2).repeat_interleave(repeats, 1)
            all_v = torch.cat(values, 2).repeat_interleave(repeats, 1)
            if self.explicit_attention:
                score = (q @ all_k.transpose(-1, -2))/math.sqrt(d//block.c.heads)
                attended = score.masked_fill(~allowed, float('-inf')).softmax(-1) @ all_v
            else:
                attended = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=allowed)
            previous.append((k, v, positions))
            h = h + block.o(attended.transpose(1, 2).reshape(b, n, d))
            z = MathBlock.rms(h, block.ffn_norm) if self.explicit_attention else block.ffn_norm(h)
            h = h + block.down(F.silu(block.gate(z))*block.up(z))
        return MathBlock.rms(h, self.norm) if self.explicit_attention else self.norm(h)

    def forward(self, tokens, budget=1, *, return_stages=False, detach_cache=False):
        check_budget(budget)
        h = self.embeddings(tokens)
        positions = torch.arange(h.shape[1], device=h.device)
        cache, outputs = [[] for _ in self.blocks], []
        for _ in range(budget+1):
            h = self._cached_stage(h, cache, positions, detach_cache=detach_cache)
            outputs.append(h)
        logits = F.linear(h, self.prediction_weight)
        return (logits, tuple(outputs)) if return_stages else logits

    def sequential_reference(self, tokens, budget=1, *, return_stages=False):
        """Token-major full recomputation: no future source or deeper past stages."""
        check_budget(budget)
        source = self.embeddings(tokens)
        inputs, outputs = [[] for _ in range(budget+1)], [[] for _ in range(budget+1)]
        for t in range(source.shape[1]):
            h = source[:, t:t+1]
            for stage in range(budget+1):
                inputs[stage].append(h)
                h = self._full_body([torch.cat(x, 1) for x in inputs[:stage+1]])[-1][:, -1:]
                outputs[stage].append(h)
        states = tuple(torch.cat(x, 1) for x in outputs)
        logits = F.linear(states[-1], self.prediction_weight)
        return (logits, states) if return_stages else logits


def matrix_work(config, length, budget, batch=1, *, full_recomputation=False):
    check_budget(budget)
    if type(length) is not int or not 0 < length <= config.max_seq or type(batch) is not int or batch < 1:
        raise ValueError('Bounded positive source length and batch required')
    d, f, layers, stages = config.dim, config.ffn_dim, config.layers, budget+1
    kv = config.kv_heads*(d//config.heads)
    projection_unit = layers*length*(4*d*d+4*d*kv+6*d*f)
    attention_unit = layers*4*length*length*d
    if full_recomputation:
        projection = projection_unit*sum(range(1, stages+1))
        attention = attention_unit*sum(s*s for s in range(1, stages+1))
    else:
        projection = projection_unit*stages
        attention = attention_unit*sum(range(1, stages+1))
    head = 2*length*d*config.vocab_size
    forward = batch*(projection+attention+head)
    return dict(forward_matmul_flops=forward, training_matmul_flops=3*forward,
                projection_forward_flops=batch*projection, attention_forward_flops=batch*attention,
                head_forward_flops=batch*head, body_executions=stages,
                cached_lower_stages=not full_recomputation,
                scope='Executed dense rectangles, FMA=2, full backward=2x forward; no elementwise, optimizer, communication or hardware speed claim.')
