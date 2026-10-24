"""CPU preparation for Parcae's diagonal injection and truncated recurrence.

This is not a reproduced Parcae model or an authorized scientific trainer.
Primary reference: Prairie et al., arXiv:2604.12946, equation3/algorithm1.
The upstream reference implementation is MIT licensed; see third_party.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DiagonalInjection(nn.Module):
    """Equal-width diagonal state decay with a learned input matrix.

    Contraction of the *linear state factor* does not establish contraction
    of the nonlinear transformer core applied after this operator.
    """

    def __init__(self, dim, initial_decay=math.sqrt(1/5)):
        super().__init__()
        if type(dim) is not int or dim <= 0 or not 0 < initial_decay < 1:
            raise ValueError('Positive width and decay strictly between zero and one required')
        delta = -math.log(initial_decay)
        self.A_log = nn.Parameter(torch.zeros(dim))
        self.dt_bias = nn.Parameter(torch.full((dim,), delta+math.log(-math.expm1(-delta))))
        self.B = nn.Parameter(torch.eye(dim))
        for p in self.parameters():
            p._no_weight_decay = True

    def forward(self, state, source):
        delta = F.softplus(self.dt_bias)
        decay = torch.exp(-delta*torch.exp(self.A_log))
        return state*decay + F.linear(source, self.B)*delta


def recurrent_tail(update, state, source, total_steps, backward_cap):
    """Run the actual full depth, differentiating only its final capped tail.

    The detached prefix changes gradients, never forward depth. Source/input
    injection still has gradients in the tail. This function handles one
    declared depth, not per-sequence routing or distributed synchronization.
    """
    if (type(total_steps) is not int or type(backward_cap) is not int
            or min(total_steps, backward_cap) < 1):
        raise ValueError('Positive integer depths required')
    tail = min(total_steps, backward_cap)
    with torch.no_grad():
        for _ in range(total_steps-tail):
            state = update(state, source)
    for _ in range(tail):
        state = update(state, source)
    return state


def injection_matrix_training_flops(dim, total_steps, backward_cap):
    """Per-position B-matrix work only: multiply/add counts as two FLOPs.

    Excludes the core, coda C matrix, vocabulary head, attention, elementwise
    operations, padding, checkpoint recomputation and optimizer. B projection
    is deliberately not hoisted out of the recurrent applications here.
    """
    if any(type(v) is not int or v < 1 for v in (dim, total_steps, backward_cap)):
        raise ValueError('Positive integer sizes required')
    return dim*dim*(2*total_steps+4*min(total_steps, backward_cap))
