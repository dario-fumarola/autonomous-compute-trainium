"""Small deterministic software arrays for sharding correctness, never data."""
import copy

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from .scale_sharded_candidate import full_parameter, local_parameter


class SoftwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(128, 64),
            adapter=nn.ModuleDict(dict(B=nn.Linear(64, 64, bias=False)))))
        self.core = nn.Linear(64, 64, bias=False)
        self.ve_gate = nn.Linear(32, 2, bias=False)
        self.norm = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 128, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if 'norm' not in name:
                    parameter.normal_(std=.05)

    def forward(self, tokens):
        source = self.transformer.wte(tokens)
        state = source
        for _ in range(2):
            gate = self.ve_gate(state[..., :32]).sigmoid().mean(-1, keepdim=True)
            state = torch.tanh(self.core(state) + self.transformer.adapter.B(source)) * gate
        return F.linear(self.norm(state), self.lm_head.weight)


def fixture_batch(rank, update, device='cpu'):
    x = (torch.arange(8).reshape(2,4) * 3 + rank * 7 + update) % 128
    y = (x * 5 + 11) % 128
    mask = ((torch.arange(8).reshape(2,4) + rank + update) % 3 != 0).float()
    return x.to(device), y.to(device), mask.to(device)


def global_count(world, update):
    return sum(int(fixture_batch(rank, update)[2].sum()) for rank in range(world))


def loss(model, rank, world, update, device):
    x, y, mask = fixture_batch(rank, update, device)
    values = F.cross_entropy(model(x).flatten(0,1), y.flatten(), reduction='none').reshape_as(mask)
    return (values * mask).sum() / global_count(world, update)


def cpu_step(model, optimizer, world, update):
    optimizer.zero_grad()
    value = 0.
    for rank in range(world):
        scalar = loss(model, rank, world, update, 'cpu')
        scalar.backward()
        value += float(scalar.detach())
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False))
    gradients = {n:p.grad.detach().clone() for n,p in model.named_parameters()}
    optimizer.step()
    return value, norm, gradients


def full_moments(optimizer):
    """All ranks participate; return a named full moment inventory on rank zero."""
    result = {}
    for parameter, alias in zip(optimizer.groups[0]['params'], optimizer.local_adam, strict=True):
        for key, value in optimizer.adamw.state[alias].items():
            name = optimizer.names[id(parameter)] + ':' + key
            if key == 'step':
                result[name] = value.detach().cpu()
            else:
                from torch.distributed.tensor import DTensor
                full = DTensor.from_local(value, parameter.device_mesh, parameter.placements,
                    shape=parameter.shape, stride=parameter.stride(), run_check=False).full_tensor()
                if optimizer.rank == 0:
                    result[name] = full.detach().cpu()
    for index, group in enumerate(optimizer.groups[1:]):
        shape = tuple(group['params'][0].shape)
        owner = index % optimizer.world
        for key in ['m', 'v']:
            if optimizer.rank == owner:
                value = optimizer.muon_state[index][key].clone()
            else:
                vshape = (len(group['params']), max(shape), 1) if shape[0]>=shape[1] else (len(group['params']),1,max(shape))
                value = torch.empty((len(group['params']),*shape) if key=='m' else vshape,
                    dtype=torch.float32,device=local_parameter(group['params'][0]).device)
            dist.broadcast(value, src=owner)
            if optimizer.rank == 0:
                result[f'muon-{index}:{key}'] = value.cpu()
    return result


def cpu_moments(optimizer, named):
    names = {id(p):n for n,p in named}
    result = {}
    for parameter, state in optimizer.adamw.state.items():
        for key, value in state.items():
            result[names[id(parameter)] + ':' + key] = value.detach().cpu()
    for i, group in enumerate(optimizer.muon.param_groups):
        for key, value in optimizer.muon.state[group['params'][0]].items():
            result[f'muon-{i}:{key}'] = value.detach().cpu()
    return result


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def relative(expected, actual):
    if set(expected) != set(actual): raise AssertionError('Named inventory changed')
    error = reference = 0.
    maximum = 0.; named_max = 0.
    for name, wanted in expected.items():
        seen = actual[name]
        if wanted.shape != seen.shape: raise AssertionError('Named tensor shape changed')
        if not torch.isfinite(wanted).all() or not torch.isfinite(seen).all():
            raise FloatingPointError('Nonfinite named numerical comparison')
        diff = (wanted.double()-seen.double())
        error += float(diff.square().sum()); reference += float(wanted.double().square().sum())
        maximum = max(maximum, float(diff.abs().max()))
        named_max = max(named_max,(float(diff.square().sum())/max(float(wanted.double().square().sum()),1e-30))**.5)
    return dict(relative_l2=(error/max(reference,1e-30))**.5, maximum_absolute_error=maximum,
        largest_named_relative_l2=named_max)


def gather_named(model, gradients=False):
    result = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad.full_tensor() if gradients else full_parameter(parameter)
        if dist.get_rank() == 0: result[name] = value.detach().cpu().clone()
    return result
