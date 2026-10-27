"""Numerical-only one-dimensional sharding candidate; not qualified training.

The gather/partial-gradient pattern follows the SimpleFSDP implementation in
AWS/Meta TorchTitan-Neuron, commit 3f447e28a4cf2f8ba939024bfdd427ee8084ee8c,
BSD-3-Clause. This small adapter keeps existing parameter names and tied aliases.
It deliberately omits TP, hybrid meshes and mixed-precision reductions.

Muon is applied to complete matrices on one owner per group. Applying Muon to
independent row shards would change its polar iteration. AdamW is elementwise
and operates on local shards. Both retain the existing FP32 mathematical recipe.
"""
import copy
import sys

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, Partial, distribute_tensor

from .parcae_optimizer_candidate import parameter_groups
from .parcae_optimizer_precise import muon_tensor_step_fp32

_WRAPPERS = {}


def _gather_parameter(module, name):
    master = module._parameters[name]
    return master.redistribute(placements=[Replicate()]).to_local(
        grad_placements=[Partial(reduce_op='sum')])


def shard_model(model, mesh):
    if mesh.ndim != 1:
        raise ValueError('Only a one-dimensional fully sharded mesh is qualified here')
    world = dist.get_world_size()
    aliases = {}
    for module in model.modules():
        names = []
        for name, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            if isinstance(parameter, DTensor) or parameter.ndim == 0:
                raise ValueError('Unwrapped nonscalar FP32 parameters required')
            if parameter.dtype != torch.float32:
                raise ValueError('FP32 master parameters required')
            key = id(parameter)
            if key not in aliases:
                placement = Shard(0) if parameter.shape[0] % world == 0 else Replicate()
                aliases[key] = nn.Parameter(distribute_tensor(parameter.detach(), mesh, [placement]))
            module._parameters[name] = aliases[key]
            names.append(name)
        if not names:
            continue

        key = (module.__class__, tuple(names))
        if key not in _WRAPPERS:
            class_name = 'ScaleShard' + module.__class__.__name__ + str(len(_WRAPPERS))
            properties = {name: property(lambda self, n=name: _gather_parameter(self, n)) for name in names}
            _WRAPPERS[key] = type(class_name, (module.__class__,), properties)
            sys.modules[__name__].__dict__[class_name] = _WRAPPERS[key]
        module.__class__ = _WRAPPERS[key]
    return model


def local_parameter(parameter):
    if not isinstance(parameter, DTensor):
        raise TypeError('Expected sharded DTensor master')
    return parameter.to_local()


def full_parameter(parameter):
    return parameter.full_tensor().detach()


class ShardedExactOptimizer:
    """Owner-only complete-matrix Muon; local elementwise AdamW.

    Checkpoints are topology-bound private qualification shards, never a legacy
    full-state scientific checkpoint. All ranks must call methods in lockstep.
    """
    def __init__(self, model, *, lr=1e-4, muon_lr=1e-4, fused_adamw=False):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.named = list(model.named_parameters())
        self.groups = parameter_groups(self.named, gate_policy='release_muon', lr=lr, muon_lr=muon_lr)
        self.param_groups = self.groups
        self.names = {id(p): name for name, p in self.named}
        self.local_adam = [nn.Parameter(local_parameter(p).detach()) for p in self.groups[0]['params']]
        adam_group = {k: v for k, v in self.groups[0].items() if k != 'params'}
        self.adamw = torch.optim.AdamW([dict(adam_group, params=self.local_adam)],
            lr=lr, betas=(.8, .95), eps=1e-10, weight_decay=0., foreach=False, fused=fused_adamw)
        self.muon_state = {}
        self.update = 0
        self.schema = dict(identity='scale_sharded_exact_fp32_muon_adamw_v1', world=self.world,
            groups=[[self.names[id(p)] for p in g['params']] for g in self.groups],
            owners=[i % self.world for i in range(len(self.groups)-1)],
            placements={n: str(p.placements[0]) for n, p in self.named})

    def zero_grad(self):
        for _, parameter in self.named:
            parameter.grad = None
        self.adamw.zero_grad(set_to_none=True)

    @torch.no_grad()
    def finish_gradients(self, *, scale=1., clip=1.):
        if scale <= 0 or clip <= 0:
            raise ValueError('Positive scale and clipping bound required')
        first = local_parameter(self.named[0][1])
        total = torch.zeros((), dtype=torch.float32, device=first.device)
        for _, parameter in self.named:
            if parameter.grad is None or not isinstance(parameter.grad, DTensor):
                raise ValueError('All masters need DTensor gradients')
            parameter.grad = parameter.grad.redistribute(placements=parameter.placements)
            gradient = parameter.grad.to_local()
            gradient.div_(scale)
            # Replicated small parameters contribute exactly once to the norm.
            factor = 1. / self.world if parameter.placements[0].is_replicate() else 1.
            total.add_(gradient.square().sum() * factor)
        dist.all_reduce(total)
        norm = total.sqrt()
        value = float(norm.cpu())
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError('Nonfinite global sharded gradient; no update')
        coefficient = (clip / (norm + 1e-6)).clamp(max=1.)
        for _, parameter in self.named:
            parameter.grad.to_local().mul_(coefficient)
        return value

    @torch.no_grad()
    def step(self):
        if any(p.grad is None for _, p in self.named):
            raise ValueError('All gradients required before either optimizer family mutates')
        for alias, parameter in zip(self.local_adam, self.groups[0]['params'], strict=True):
            alias.grad = parameter.grad.to_local()
        for key, value in self.groups[0].items():
            if key != 'params':
                self.adamw.param_groups[0][key] = value
        self.adamw.step()
        for index, group in enumerate(self.groups[1:]):
            owner = index % self.world
            parameters = group['params']
            packed = torch.stack([full_parameter(p) for p in parameters])
            gradients = torch.stack([p.grad.full_tensor() for p in parameters])
            shape = tuple(parameters[0].shape)
            reduction = -1 if shape[-2] >= shape[-1] else -2
            if self.rank == owner:
                if index not in self.muon_state:
                    vshape = (len(parameters), shape[-2], 1) if reduction == -1 else (len(parameters), 1, shape[-1])
                    self.muon_state[index] = dict(m=torch.zeros_like(packed),
                        v=torch.zeros(vshape, device=packed.device, dtype=torch.float32))
                state = self.muon_state[index]
                scalars = torch.tensor([group['momentum'], group['lr'] * max(1., shape[-2]/shape[-1])**.5,
                    group['weight_decay'], group.get('beta2', .95)], dtype=torch.float32).to(packed.device)
                muon_tensor_step_fp32(gradients, packed, state['m'], state['v'],
                    *scalars.unbind(0), group['ns_steps'], reduction)
            dist.broadcast(packed, src=owner)
            for parameter, value in zip(parameters, packed.unbind(0), strict=True):
                if parameter.placements[0].is_shard():
                    value = value.chunk(self.world, dim=0)[self.rank]
                local_parameter(parameter).copy_(value)
        self.update += 1

    def state_dict(self):
        return dict(purpose='scale_sharded_qualification_only', qualification_only=True,
            eligible_for_science=False, rank=self.rank, schema=copy.deepcopy(self.schema), update=self.update,
            adamw=self.adamw.state_dict(), muon=self.muon_state,
            groups=[{k:v for k,v in g.items() if k != 'params'} for g in self.groups])

    def load_state_dict(self, state):
        if (state['purpose'] != 'scale_sharded_qualification_only' or state['eligible_for_science'] is not False
                or state['schema'] != self.schema or state['rank'] != self.rank):
            raise ValueError('Exact same-rank qualification schema required')
        expected = {i for i in range(len(self.groups)-1) if i % self.world == self.rank}
        if set(state['muon']) != (expected if state['update'] else set()):
            raise ValueError('Missing or unexpected owner moments')
        self.adamw.load_state_dict(state['adamw'])
        device = local_parameter(self.named[0][1]).device
        self.muon_state = {i:{n:t.to(device) for n,t in row.items()} for i,row in state['muon'].items()}
        self.update = state['update']
        for group, values in zip(self.groups, state['groups'], strict=True):
            group.update(values)


def scale_inventory():
    """Meta tensors only; includes distinct optimizer moment layouts."""
    from .parcae_candidate import ParcaeCandidateConfig, ParcaeCandidate, matmul_ledger
    rows = []
    for role, recurrent, cfg in [
        ('fixed_two', True, ParcaeCandidateConfig(dim=2816, heads=22, ffn_dim=11264, max_seq=1024)),
        ('ordinary8', False, ParcaeCandidateConfig(dim=2304, heads=18, ffn_dim=11008, prelude=2, core=4, coda=2, max_seq=1024)),
    ]:
        with torch.device('meta'):
            model = ParcaeCandidate(cfg, recurrent=recurrent, initialize=False)
        groups = parameter_groups(model.named_parameters(), gate_policy='release_muon')
        adam = sum(p.numel() for p in groups[0]['params'])
        muon = sum(p.numel() for g in groups[1:] for p in g['params'])
        variance = sum(len(g['params']) * max(g['params'][0].shape) for g in groups[1:])
        rows.append(dict(role=role, config=vars(cfg), parameters=cfg.parameter_count(recurrent),
            adamw_parameters=adam, muon_parameters=muon, muon_variance_elements=variance,
            master_gradient_moment_bytes=16*adam+12*muon+4*variance,
            neural_work_per_window=matmul_ledger(cfg,1024,1,recurrent=recurrent,
                total_steps=2 if recurrent else 1,backward_cap=2 if recurrent else 1)['training_matmul_flops']))
    return rows
