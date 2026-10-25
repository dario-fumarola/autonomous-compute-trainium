"""Exact sparse-label BCE for the frozen future-span objective.

This changes loss evaluation,not the target,head,horizon or weight. Avoid a
dense vocabulary-sized target/scatter: sum softplus over the vocabulary and
subtract each distinct positive logit exactly once. Native qualification is
still required; the earlier dense CPU screen remains unchanged.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class SparseFutureTargets:
    token_ids: torch.Tensor
    distinct: torch.Tensor
    valid: torch.Tensor


def sparse_targets(targets):
    if targets.token_ids.device.type!='cpu' or targets.valid.device.type!='cpu':
        raise ValueError('Deduplicate original source labels on CPU before transfer')
    ids=targets.token_ids.sort(-1).values
    distinct=torch.cat([torch.ones_like(ids[:,:,:1],dtype=torch.bool),ids[:,:,1:]!=ids[:,:,:-1]],-1)
    return SparseFutureTargets(ids,distinct,targets.valid)


def sparse_future_loss_sum(logits, token_ids, distinct, valid, horizon):
    if (logits.ndim!=3 or token_ids.shape!=(*logits.shape[:2],horizon-1)
            or distinct.shape!=token_ids.shape or valid.shape!=logits.shape[:2]):
        raise ValueError('Sparse presence target dimensions differ')
    z=logits.float()
    positive=(z.gather(-1,token_ids)*distinct).sum(-1)
    return ((F.softplus(z).sum(-1)-positive)*valid).sum()/(horizon-1)
