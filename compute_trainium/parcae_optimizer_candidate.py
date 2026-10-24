"""Explicit release-style Muon/AdamW candidate; no active trainer imports this.

Muon tensor update derived from the MIT sandyresearch/parcae release69284c1,
copyright(c)2026 Hayden Prairie. License: third_party/parcae_reference/LICENSE.
The AdamW path uses PyTorch; gate grouping must be chosen explicitly. Native
updates, recovery and performance require their own qualification.
"""
import copy

import torch


POLAR_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, .5000178451051316),
    (3.8916678022926607, -2.772484153217685, .5060648178503393),
    (3.285753657755655, -2.3681294933425376, .46449024233003106),
    (2.3465413258596377, -1.7097828382687081, .42323551169305323),
)


def muon_tensor_step(gradient, parameters, momentum_state, variance_state,
                     momentum, lr, decay, beta2, steps, reduction_dimension):
    """Specified BF16 polar iteration with FP32 parameters/moment states.

    Scalar tensors, not changing Python learning-rate constants, are inputs to
    a future compiled graph. The BF16 casts are part of this defined optimizer.
    """
    momentum_value = momentum.to(gradient.dtype)
    momentum_state.lerp_(gradient, 1-momentum_value)
    update = gradient.lerp_(momentum_state, momentum_value)
    polar = update.bfloat16()
    polar = polar/(polar.norm(dim=(-2, -1), keepdim=True)*1.02+1e-6)
    for a, b, c in POLAR_COEFFS[:steps]:
        if gradient.size(-2) > gradient.size(-1):
            gram = polar.mT@polar
            polar = a*polar+polar@(b*gram+c*(gram@gram))
        else:
            gram = polar@polar.mT
            polar = a*polar+(b*gram+c*(gram@gram))@polar
    beta = beta2.to(polar.dtype)
    mean = polar.float().square().mean(dim=reduction_dimension, keepdim=True)
    norm = (mean.sum(dim=(-2, -1), keepdim=True)*polar.size(reduction_dimension)).sqrt()
    variance_state.lerp_(mean.to(variance_state.dtype), (1-beta).to(variance_state.dtype))
    scale = variance_state.clamp_min(1e-10).rsqrt()
    new_norm = ((mean*polar.size(reduction_dimension))*scale.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    update = polar*(scale*(norm/new_norm.clamp_min(1e-10))).to(polar.dtype)
    rate, wd = lr.to(update.dtype), decay.to(update.dtype)
    cautious = (update*parameters) >= 0
    parameters.sub_(rate*update+rate*wd*parameters*cautious)


def parameter_groups(named_parameters, *, gate_policy, lr=.008, muon_lr=.008,
                     momentum=.95, muon_decay=.2, steps=5):
    if gate_policy not in ('release_muon', 'paper_adamw'):
        raise ValueError('Choose release_muon or paper_adamw explicitly')
    if min(lr, muon_lr) <= 0 or not 1 <= steps <= len(POLAR_COEFFS):
        raise ValueError('Positive learning rates and one to five polar steps required')
    named = list(named_parameters)
    adamw, matrices = [], {}
    for name, p in named:
        if not p.requires_grad or p.dtype != torch.float32:
            raise ValueError('All master parameters must be trainable FP32')
        lower = name.lower()
        embedding = any(s in lower for s in ('wte', 'embedding', 'embed', 'lm_head'))
        normalization = any(s in lower for s in ('norm', 'ln_', 'bias'))
        injection = any(s in lower for s in ('a_log', 'dt_bias', 'adapter.b', 'adapter.adapter', 'transformer.c.'))
        paper_gate = gate_policy == 'paper_adamw' and 've_gate' in lower
        if p.ndim < 2 or embedding or normalization or injection or paper_gate:
            adamw.append(p)
        else:
            if p.ndim != 2:
                raise ValueError('Only matrix parameters are supported in Muon')
            matrices.setdefault(tuple(p.shape), []).append(p)
    groups = [{'params': adamw, 'kind': 'adamw', 'lr': lr, 'base_lr': 1.,
               'betas': (.8, .95), 'eps': 1e-10, 'weight_decay': 0.}]
    groups += [{'params': values, 'kind': 'muon', 'lr': muon_lr, 'base_lr': muon_lr/lr,
                'momentum': momentum, 'weight_decay': muon_decay, 'ns_steps': steps}
               for values in matrices.values()]
    actual = [id(p) for group in groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != {id(p) for _,p in named}:
        raise ValueError('Grouping duplicated or omitted a master parameter')
    return groups


class MuonMatrixOptimizer(torch.optim.Optimizer):
    def __init__(self, groups, *, backend=None):
        super().__init__(groups, defaults={})
        self.tensor_step = (torch.compile(muon_tensor_step, backend=backend, fullgraph=True, dynamic=False)
                            if backend else muon_tensor_step)
        # Recreated scalar buffers contain no history; moments are in state_dict.
        self._scalars = {}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            values = group['params']
            first = values[0]
            if any(p.grad is None for p in values):
                raise ValueError('Missing Muon gradient; caller must not omit a path')
            shape = tuple(first.shape)
            reduction = -1 if shape[-2] >= shape[-1] else -2
            state = self.state[first]
            if not state:
                state['m'] = torch.zeros((len(values), *shape), device=first.device, dtype=torch.float32)
                vshape = (len(values), shape[-2], 1) if reduction == -1 else (len(values), 1, shape[-1])
                state['v'] = torch.zeros(vshape, device=first.device, dtype=torch.float32)
            if first.device not in self._scalars:
                self._scalars[first.device] = {k:torch.zeros((), dtype=torch.float32, device=first.device)
                    for k in ('momentum','lr','decay','beta2')}
            buffers = self._scalars[first.device]
            buffers['momentum'].fill_(group['momentum'])
            buffers['lr'].fill_(group['lr']*max(1., shape[-2]/shape[-1])**.5)
            buffers['decay'].fill_(group['weight_decay'])
            buffers['beta2'].fill_(group.get('beta2', .95))
            gradients = torch.stack([p.grad for p in values])
            parameters = torch.stack(values)
            self.tensor_step(gradients, parameters, state['m'], state['v'],
                buffers['momentum'], buffers['lr'], buffers['decay'], buffers['beta2'],
                group['ns_steps'], reduction)
            for p, updated in zip(values, parameters.unbind(0), strict=True):
                p.copy_(updated)


class ParcaeOptimizerCandidate:
    """Complete, serializable two-family optimizer with explicit recipe metadata.

    Caller must validate global gradient finiteness and unscale before clipping
    and step. No optional fused AMP metadata or implicit gradient scaling.
    """
    def __init__(self, model, *, gate_policy, lr=.008, muon_lr=.008,
                 fused_adamw=False, muon_backend=None):
        named = list(model.named_parameters())
        groups = parameter_groups(named, gate_policy=gate_policy, lr=lr, muon_lr=muon_lr)
        self.adamw = torch.optim.AdamW([groups[0]], lr=lr, betas=(.8,.95), eps=1e-10,
                                      weight_decay=0., foreach=False, fused=fused_adamw)
        self.muon = MuonMatrixOptimizer(groups[1:], backend=muon_backend)
        self.param_groups = self.adamw.param_groups+self.muon.param_groups
        lookup = {id(p):name for name,p in named}
        self.recipe = {'identity':'parcae_muon_adamw_candidate_v1','gate_policy':gate_policy,
            'groups': [[lookup[id(p)] for p in g['params']] for g in self.param_groups],
            'specified_muon_polar_dtype':'bf16','master_and_moment_dtype':'fp32'}

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self):
        # Check every group before either family mutates model/state.
        if any(p.grad is None for g in self.param_groups for p in g['params']):
            raise ValueError('Missing gradient; no optimizer family has stepped')
        self.adamw.step()
        self.muon.step()

    def state_dict(self):
        return {'recipe':copy.deepcopy(self.recipe), 'adamw':self.adamw.state_dict(),
                'muon':self.muon.state_dict()}

    def load_state_dict(self, state):
        if state['recipe'] != self.recipe:
            raise ValueError('Optimizer grouping or specified recipe changed')
        self.adamw.load_state_dict(state['adamw'])
        self.muon.load_state_dict(state['muon'])
        self.param_groups = self.adamw.param_groups+self.muon.param_groups
