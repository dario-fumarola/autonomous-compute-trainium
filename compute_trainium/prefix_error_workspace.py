"""Temporary vocabulary-bias correction from already observed prediction errors.

This is a diagnostic of causal dynamic evaluation, not a trained thought writer.
The caller supplies strictly past-only weights; current labels never enter a read.
"""
import torch
from torch.nn import functional as F
from compute_trainium.causal_associative_workspace import prefix_weights


def read_weights(features, anchors, *, temperature=16.):
    unit = F.normalize(features, dim=-1)
    return prefix_weights(
        (unit.index_select(1, anchors) @ unit.transpose(-1, -2)) * temperature,
        anchors,
    ).to(features.dtype)


def observed_errors(logits, outcomes):
    """Negative CE bias gradient, avoiding 1-p_y cancellation near p_y=1."""
    # Small vocabulary-bias changes require double-precision normalization and
    # accumulation. At the real 100352-class shape, FP32 exceeded the registered
    # .0002 target-logp tolerance at the largest frozen step size.
    probability = logits.double().softmax(-1)
    others = probability.scatter(-1, outcomes[..., None], 0.)
    return (-others).scatter(-1, outcomes[..., None], others.sum(-1, keepdim=True))


def read_correction(logits, outcomes, weights, *, mode='error'):
    weights = weights.double()
    if mode == 'error':
        return weights @ observed_errors(logits, outcomes)
    if mode == 'prediction':
        return -(weights @ logits.double().softmax(-1))
    if mode == 'label':
        output = weights.new_zeros(*weights.shape[:-1], logits.shape[-1])
        indices = outcomes[:, None, :].expand(-1, weights.shape[1], -1)
        return output.scatter_add(-1, indices, weights)
    raise ValueError('Unknown correction control')


def rotated_error(logits, outcomes, weights, anchors):
    """Rotate complete observed error vectors within each available prefix."""
    errors = observed_errors(logits, outcomes)
    weights = weights.double()
    output = weights.new_zeros(logits.shape[0], len(anchors), logits.shape[-1])
    for qi, end in enumerate(anchors.tolist()):
        if end:
            output[:, qi] = (weights[:, qi:qi+1, :end] @
                             errors[:, :end].roll(1, dims=1)).squeeze(1)
    return output


def sequential_bias_reference(logits, outcomes, weights, anchors):
    """Independent literal negative gradient of each weighted prefix CE."""
    rows = []
    for qi, end in enumerate(anchors.tolist()):
        bias = torch.zeros(logits.shape[0], logits.shape[-1],
                           dtype=logits.dtype, requires_grad=True)
        if not end:
            rows.append(bias.detach())
            continue
        shifted = logits[:, :end].detach() + bias[:, None, :]
        loss = F.cross_entropy(shifted.flatten(0, 1), outcomes[:, :end].flatten(),
                               reduction='none').reshape(logits.shape[0], end)
        gradient = torch.autograd.grad((loss * weights[:, qi, :end]).sum(), bias)[0]
        rows.append(-gradient.detach())
    return torch.stack(rows, 1)
