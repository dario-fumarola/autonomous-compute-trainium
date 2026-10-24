"""Static PyTorch model candidate, checked against the MIT Parcae release.

Reference: sandyresearch/parcae at69284c1. Copyright(c)2026 Hayden Prairie;
license is preserved in third_party/parcae_reference/LICENSE. This module is
not imported by any current scientific trainer. Native execution, optimizer
and sampled-depth training remain separate qualification requirements.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .stable_recurrence_candidate import DiagonalInjection, recurrent_tail


@dataclass(frozen=True)
class ParcaeCandidateConfig:
    dim: int = 768
    heads: int = 6
    ffn_dim: int = 3072
    vocab_size: int = 32768
    prelude: int = 2
    core: int = 2
    coda: int = 2
    max_seq: int = 2048
    rope_base: float = 50000.
    norm_eps: float = 1e-5

    def __post_init__(self):
        if (min(self.dim, self.heads, self.ffn_dim, self.vocab_size, self.prelude,
                self.core, self.coda, self.max_seq) < 1 or self.dim < 32
                or self.dim % self.heads or (self.dim//self.heads) % 2
                or self.core % 2 or self.norm_eps <= 0):
            raise ValueError('Valid equal-width MHA dimensions and an even-sized core required')

    @property
    def layers(self):
        return self.prelude+self.core+self.coda

    def has_value_embedding(self, layer):
        return layer % 2 == (self.layers-1) % 2

    def parameter_count(self, recurrent=True):
        tables = 1+sum(self.has_value_embedding(i) for i in range(self.layers))
        block = 4*self.dim**2+2*self.dim*self.ffn_dim+2*self.dim
        gates = sum(self.has_value_embedding(i) for i in range(self.layers))*32*self.heads
        ordinary = tables*self.vocab_size*self.dim+self.layers*block+self.dim+gates
        return ordinary+(2*self.dim**2+3*self.dim if recurrent else 0)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        z = x.float()
        return (z*torch.rsqrt(z.square().mean(-1, keepdim=True)+self.eps)).to(x.dtype)*self.weight


class Attention(nn.Module):
    def __init__(self, config, value_embedding):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.dim//config.heads
        self.c_q = nn.Linear(config.dim, config.dim, bias=False)
        self.c_k = nn.Linear(config.dim, config.dim, bias=False)
        self.c_v = nn.Linear(config.dim, config.dim, bias=False)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.ve_gate = nn.Linear(32, config.heads, bias=False) if value_embedding else None

    def forward(self, x, freqs, ve=None):
        b, n, width = x.shape
        shape = (b, n, self.heads, self.head_dim)
        q, k, v = self.c_q(x).view(shape), self.c_k(x).view(shape), self.c_v(x).view(shape)
        if ve is not None:
            gate = 2*torch.sigmoid(self.ve_gate(x[..., :32]))
            v = v+gate[..., None]*ve.view(shape)
        # Pairwise real rotation, matching the release's FP32 RoPE arithmetic.
        pair = torch.cat([q, k], dim=2).unflatten(-1, (-1, 2)).float()
        rot = torch.stack([pair[..., 0]*freqs[..., 0]-pair[..., 1]*freqs[..., 1],
                           pair[..., 1]*freqs[..., 0]+pair[..., 0]*freqs[..., 1]], -1).flatten(3)
        q, k = torch.split(rot.to(q.dtype), self.heads, dim=2)
        q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2), is_causal=True).transpose(1, 2)
        return self.c_proj(y.contiguous().view(b, n, width))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.proj = nn.Linear(config.ffn_dim, config.dim, bias=False)

    def forward(self, x):
        # The pinned release scales squared ReLU by0.5.
        return self.proj(F.relu(self.fc(x)).square()*.5)


class Block(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.norm_1 = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config, config.has_value_embedding(layer))
        self.norm_2 = RMSNorm(config.dim, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs, ve=None):
        x = x+self.attn(self.norm_1(x), freqs, ve)
        return x+self.mlp(self.norm_2(x))


class ParcaeCandidate(nn.Module):
    """Fixed-shape forward with explicit initial state and backward depth.

    Supplied state makes CPU/native replay and noise accounting explicit.
    Logical total/backward depth are Python integers to be frozen per graph.
    No router, sampler, cache, teacher, optimizer or auxiliary objective.
    """
    def __init__(self, config, *, recurrent=True, initialize=True):
        super().__init__()
        self.config, self.recurrent = config, recurrent
        table = {'wte': nn.Embedding(config.vocab_size, config.dim)}
        if recurrent:
            offset = 0
            for name, length in [('prelude', config.prelude), ('core_block', config.core), ('coda', config.coda)]:
                table[name] = nn.ModuleList(Block(config, i) for i in range(offset, offset+length))
                offset += length
            table.update(adapter=DiagonalInjection(config.dim),
                C=nn.Linear(config.dim, config.dim, bias=False),
                ln_prelude=RMSNorm(config.dim, config.norm_eps))
        else:
            table['h'] = nn.ModuleList(Block(config, i) for i in range(config.layers))
        table['ln_f'] = RMSNorm(config.dim, config.norm_eps)
        self.transformer = nn.ModuleDict(table)
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(config.vocab_size, config.dim)
            for i in range(config.layers) if config.has_value_embedding(i)})
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        frequency = config.rope_base**(-torch.arange(0, config.dim//config.heads, 2).float()/(config.dim//config.heads))
        phase = torch.outer(torch.arange(config.max_seq).float(), frequency)
        self.register_buffer('freqs_cis', torch.stack([phase.cos()[None, :, None, :],
                                                     phase.sin()[None, :, None, :]], dim=4))
        if initialize:
            self.reset_parameters()
        if sum(p.numel() for p in self.parameters()) != config.parameter_count(recurrent):
            raise AssertionError('Operator inventory and parameter count disagree')

    def reset_parameters(self):
        """Release initialization distributions; not its identical RNG draw order."""
        d = self.config.dim
        std = math.sqrt(2/(5*d))
        with torch.no_grad():
            nn.init.trunc_normal_(self.transformer.wte.weight, std=std, a=-3*std, b=3*std)
            for embedding in self.value_embeds.values():
                nn.init.normal_(embedding.weight)
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)
                if isinstance(module, Block):
                    if module.attn.ve_gate is not None:
                        module.attn.ve_gate.reset_parameters()
                    for matrix in [module.attn.c_q.weight, module.attn.c_k.weight,
                                   module.attn.c_v.weight, module.mlp.fc.weight]:
                        rows, cols = matrix.shape
                        noise = torch.empty((max(rows, cols), min(rows, cols)), device=matrix.device, dtype=matrix.dtype)
                        nn.init.trunc_normal_(noise, std=1.)
                        q, r = torch.linalg.qr(noise)
                        q *= torch.diag(r).sign()
                        matrix.copy_((q if rows >= cols else q.T)*(std*math.sqrt(max(rows, cols))))
                    nn.init.zeros_(module.attn.c_proj.weight)
                    nn.init.zeros_(module.mlp.proj.weight)
            if self.recurrent:
                self.transformer.adapter.A_log.zero_()
                delta = -math.log(math.sqrt(1/5))
                self.transformer.adapter.dt_bias.fill_(delta+math.log(-math.expm1(-delta)))
                self.transformer.adapter.B.copy_(torch.eye(d, device=self.transformer.adapter.B.device))
                # C follows the same orthogonal initialization law as input maps.
                m = self.transformer.C.weight
                noise = torch.empty_like(m)
                nn.init.trunc_normal_(noise, std=1.)
                q, r = torch.linalg.qr(noise)
                m.copy_(q*torch.diag(r).sign()*math.sqrt(2/5))

    def initial_state(self, tokens, generator=None):
        state = torch.empty((*tokens.shape, self.config.dim), device=tokens.device,
                            dtype=self.transformer.wte.weight.dtype)
        std = math.sqrt(2/(5*self.config.dim))
        return nn.init.trunc_normal_(state, std=std, a=-3*std, b=3*std, generator=generator)

    def hidden(self, tokens, state=None, *, total_steps=1, backward_cap=1):
        if tokens.shape[1] > self.config.max_seq:
            raise ValueError('Input exceeds declared rotary context')
        freqs = self.freqs_cis[:, :tokens.shape[1]]
        e = self.transformer.wte(tokens)
        def block(block, x, i):
            ve = self.value_embeds[str(i)](tokens) if str(i) in self.value_embeds else None
            return block(x, freqs, ve)
        if not self.recurrent:
            if total_steps != 1 or state is not None:
                raise ValueError('The ordinary reference executes each block once')
            x = e
            for i, layer in enumerate(self.transformer.h):
                x = block(layer, x, i)
        else:
            if state is None:
                raise ValueError('Explicit random initial state required for recurrence')
            if state.shape != (*tokens.shape, self.config.dim):
                raise ValueError('Initial state shape differs from the current source window')
            for i, layer in enumerate(self.transformer.prelude):
                e = block(layer, e, i)
            e = self.transformer.ln_prelude(e)
            def update(x, source):
                x = self.transformer.adapter(x, source)
                for j, layer in enumerate(self.transformer.core_block):
                    x = block(layer, x, self.config.prelude+j)
                return x
            x = recurrent_tail(update, state, e, total_steps, backward_cap)
            x = self.transformer.C(x)
            for j, layer in enumerate(self.transformer.coda):
                x = block(layer, x, self.config.prelude+self.config.core+j)
        return self.transformer.ln_f(x)

    def forward(self, tokens, state=None, *, total_steps=1, backward_cap=1):
        return F.linear(self.hidden(tokens, state, total_steps=total_steps,
                                    backward_cap=backward_cap), self.lm_head.weight).float()


def matmul_ledger(config, positions, batch=1, *, recurrent=True, total_steps=1, backward_cap=1):
    """Executed dense matmuls with explicit forward-only and backward depths.

    Full-square attention, FMA=2; no activation recomputation. Embedding gathers,
    norms, elementwise operators, optimizer, distributed work and padding outside
    the declared dense shapes are separate. The tied output head is counted.
    """
    if any(type(v) is not int or v < 1 for v in (positions, batch, total_steps, backward_cap)):
        raise ValueError('Positive integer shapes/depths required')
    if not recurrent and total_steps != 1:
        raise ValueError('Ordinary graph has one pass')
    d, n, f = config.dim, positions, config.ffn_dim
    dense = batch*(n*(8*d*d+4*d*f)+4*n*n*d)
    gate = batch*2*n*32*config.heads
    def blocks(indices):
        return sum(dense+(gate if config.has_value_embedding(i) else 0) for i in indices)
    head = batch*2*n*d*config.vocab_size
    if recurrent:
        prelude = blocks(range(config.prelude))
        core = blocks(range(config.prelude, config.prelude+config.core))
        coda = blocks(range(config.prelude+config.core, config.layers))
        projection = batch*2*n*d*d
        once = prelude+coda+projection+head  # output C projection
        repeated = core+projection  # input B projection on every application
        tail = min(total_steps, backward_cap)
        forward = once+repeated*total_steps
        backward = 2*once+2*repeated*tail
    else:
        forward = blocks(range(config.layers))+head
        backward = 2*forward
    return {'forward_matmul_flops': forward, 'backward_matmul_flops': backward,
            'training_matmul_flops': forward+backward,
            'head_forward_matmul_flops': head,
            'scope': 'Declared dense executed graph; embedding gathers and elementwise/optimizer/distributed costs excluded. No hardware latency claim.'}
