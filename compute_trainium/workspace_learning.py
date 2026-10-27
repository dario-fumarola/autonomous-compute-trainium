"""Uncached causal reference for sampled workspaces and detached policy learning.

Token arrays used by tests are software fixtures, never research examples.
The writer API intentionally has no future-target argument.
"""
import copy
import math
from dataclasses import dataclass, field
import time

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WorkspaceFormat:
    opening: tuple[int, ...] = ()
    closing: tuple[int, ...] = ()


@dataclass
class ForwardLedger:
    events: list = field(default_factory=list)

    def summary(self):
        return dict(forward_calls=len(self.events),
                    processed_positions=sum(e['batch']*e['input_positions'] for e in self.events),
                    projected_positions=sum(e['batch']*e['output_positions'] for e in self.events),
                    forward_wall_seconds=sum(e['wall_seconds'] for e in self.events),
                    events=self.events,
                    warning='Full recomputation reference. Positions are not FLOPs. Backward,optimizer,EMA and sampling costs need separate measurement.')


class Reader:
    """A callable returning full causal logits; never retains a K/V cache."""
    def __init__(self, logits, ledger=None):
        self.logits=logits;self.ledger=ledger

    def __call__(self,tokens,phase):
        _tokens(tokens,'model input');started=time.perf_counter()
        value=self.logits(tokens)
        if (not isinstance(value,torch.Tensor) or value.ndim!=3
                or value.shape[:2]!=tokens.shape):
            raise ValueError('Reference reader requires full batch/position/vocabulary logits')
        if self.ledger is not None:
            self.ledger.events.append(dict(phase=phase,batch=tokens.shape[0],input_positions=tokens.shape[1],
                output_positions=value.shape[1],vocabulary=value.shape[2],wall_seconds=time.perf_counter()-started))
        return value


def _tokens(value,name,*,empty=False):
    if (not isinstance(value,torch.Tensor) or value.dtype!=torch.long or value.ndim!=2
            or value.shape[0]<1 or (not empty and value.shape[1]<1)):
        raise ValueError(name+' must be a nonempty-batch rank-two int64 tensor')


def _constant(tokens,values):
    return tokens.new_tensor(values).reshape(1,-1).expand(tokens.shape[0],-1)


def framed_context(prefix,workspace,format):
    _tokens(prefix,'prefix');_tokens(workspace,'workspace',empty=True)
    if prefix.shape[0] not in [1,workspace.shape[0]]:raise ValueError('Incompatible context batch')
    prefix=prefix.expand(workspace.shape[0],-1)
    return torch.cat([prefix,_constant(prefix,format.opening),workspace,_constant(prefix,format.closing)],dim=1)


def target_log_probs(reader,context,targets,*,phase,temperature=1.):
    _tokens(context,'context');_tokens(targets,'targets')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite sampling temperature required')
    if targets.shape[0] not in [1,context.shape[0]]:raise ValueError('Incompatible target batch')
    targets=targets.expand(context.shape[0],-1)
    inputs=torch.cat([context,targets[:,:-1]],dim=1)
    logits=reader(inputs,phase)[:,context.shape[1]-1:,:].float()/temperature
    return F.log_softmax(logits,dim=-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def sample_workspaces(reader,prefix,*,format,length,group_size,generator,temperature=1.):
    _tokens(prefix,'prefix')
    if prefix.shape[0]!=1 or type(length) is not int or not 0<=length<=256:
        raise ValueError('One prefix and bounded nonnegative length required')
    if type(group_size) is not int or not 2<=group_size<=32 or not 0<temperature<=10:
        raise ValueError('Bounded independent group and temperature required')
    context=torch.cat([prefix,_constant(prefix,format.opening)],dim=1).expand(group_size,-1)
    samples=[]
    for _ in range(length):
        probabilities=torch.softmax(reader(context,'workspace_sampling')[:,-1,:].float()/temperature,dim=-1)
        token=torch.multinomial(probabilities,1,generator=generator)
        samples.append(token);context=torch.cat([context,token],dim=1)
    return torch.cat(samples,dim=1) if samples else prefix.new_empty((group_size,0))


@torch.no_grad()
def score_workspaces(scorer,prefix,targets,workspace,*,format,cost_per_token=0.):
    if not math.isfinite(cost_per_token) or cost_per_token<0:raise ValueError('Nonnegative finite cost required')
    raw=target_log_probs(scorer,prefix,targets,phase='scorer_raw_no_workspace').mean(-1)
    framed=target_log_probs(scorer,framed_context(prefix,workspace[:1,:0],format),targets,
                            phase='scorer_empty_frame').mean(-1)
    worked=target_log_probs(scorer,framed_context(prefix,workspace,format),targets,
                            phase='scorer_workspace').mean(-1)
    gain=worked-raw
    return dict(rewards=(gain-cost_per_token*workspace.shape[1]).detach(),
                absolute_gain=gain.detach(),framed_gain=(worked-framed).detach(),
                raw_log_probability=raw.detach(),framed_log_probability=framed.detach(),
                workspace_log_probability=worked.detach())


def group_advantages(rewards):
    if rewards.ndim!=1 or rewards.numel()<2 or not torch.isfinite(rewards).all():
        raise ValueError('At least two finite scalar rewards required')
    values=rewards.detach()
    return (values-values.mean())*(values.numel()/(values.numel()-1))


def policy_loss(actor,prefix,workspace,rewards,*,format,temperature=1.):
    _tokens(workspace,'policy workspace')
    if rewards.shape!=(workspace.shape[0],):raise ValueError('One reward per sampled workspace required')
    context=torch.cat([prefix,_constant(prefix,format.opening)],dim=1).expand(workspace.shape[0],-1)
    log_probs=target_log_probs(actor,context,workspace,phase='actor_policy',temperature=temperature)
    # Sum over actions: averaging by length would change the sequence objective.
    return -(group_advantages(rewards)*log_probs.sum(-1)).mean()


def reader_loss(actor,prefix,targets,workspace,*,format):
    return -target_log_probs(actor,framed_context(prefix,workspace,format),targets,
                             phase='actor_workspace_reader').mean()


def make_frozen_scorer(actor):
    return copy.deepcopy(actor).eval().requires_grad_(False)


@torch.no_grad()
def update_ema(scorer,actor,*,decay):
    if not 0<=decay<1:raise ValueError('EMA decay in [0,1) required')
    left=dict(scorer.named_parameters());right=dict(actor.named_parameters())
    if left.keys()!=right.keys():raise ValueError('Actor and scorer parameters differ')
    lb=dict(scorer.named_buffers());rb=dict(actor.named_buffers())
    if lb.keys()!=rb.keys():raise ValueError('Actor and scorer buffers differ')
    for key,p in left.items():
        if p.shape!=right[key].shape or p.data_ptr()==right[key].data_ptr():
            raise ValueError('Separate matching scorer parameters required')
        p.mul_(decay).add_(right[key].detach(),alpha=1-decay)
    for key,b in lb.items():b.copy_(rb[key])
