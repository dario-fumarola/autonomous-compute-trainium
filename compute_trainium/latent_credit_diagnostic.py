"""CPU gradient diagnostic, not an admitted training recipe.

Forward predictions are identical; stopping gradients changes the learning rule.
The retained refinement count is not an inference thought budget.
"""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F


def jacobi_credit(model, tokens, refinements, retained, *, return_state=False):
    if type(refinements) is not int or refinements < 1:
        raise ValueError('Positive numerical refinement count required')
    if retained != 'all' and (type(retained) is not int or not 0 <= retained < refinements):
        raise ValueError('All gradients or a strict suffix of refinements required')
    source = model.embeddings(tokens)
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    with nullcontext() if retained == 'all' else torch.no_grad():
        state = model.body_at(source, positions)
    for index in range(refinements):
        active = retained == 'all' or index >= refinements-retained
        with nullcontext() if active else torch.no_grad():
            state = model.body_at(model.interleave(source, state),
                                  positions.repeat_interleave(2))[:, ::2]
    final = model.body_at(model.interleave(source, state), positions.repeat_interleave(2))[:, 1::2]
    logits = F.linear(final, model.prediction_weight)
    return (logits, state) if return_state else logits


def matrix_work(config, length, refinements, retained, batch=1):
    if type(refinements) is not int or refinements < 1:
        raise ValueError('Positive numerical refinement count required')
    if retained != 'all' and (type(retained) is not int or not 0 <= retained < refinements):
        raise ValueError('All gradients or strict refinement suffix')
    initial = config.flops(0, length, batch)['forward_matmul_flops']
    expanded = config.flops(0, 2*length, batch)['forward_matmul_flops']
    head = 2*batch*length*config.dim*config.vocab_size
    forward = initial+(refinements+1)*expanded+head
    differentiated = forward if retained == 'all' else (retained+1)*expanded+head
    return dict(forward_matmul_flops=forward, differentiated_forward_matmul_flops=differentiated,
                training_matmul_flops=forward+2*differentiated,
                scope='Analytic matrix ledger only; same forwards, fewer backwards. '
                      'Not a trained quality result, measured runtime or unchanged gradient.')


def compare_gradients(reference, candidate):
    if set(reference) != set(candidate) or not reference:
        raise ValueError('Matching complete gradient maps required')
    totals = {k: [0., 0., 0., 0.] for k in ('all', 'embedding', 'body', 'head')}
    for name, left in reference.items():
        right = candidate[name]
        if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError('Finite shape-matched gradients required')
        family = 'embedding' if name.startswith('embedding.') else 'head' if name == 'prediction_weight' else 'body'
        left, right = left.double(), right.double()
        values = [float(left.square().sum()), float(right.square().sum()),
                  float((left*right).sum()), float((right-left).square().sum())]
        for key in ('all', family):
            totals[key] = [a+b for a, b in zip(totals[key], values)]
    result = {}
    for key, (aa, bb, ab, dd) in totals.items():
        if aa <= 0 or bb <= 0:
            raise ValueError('Each registered gradient family must be nonzero')
        result[key] = dict(cosine=ab/math.sqrt(aa*bb), relative_l2_error=math.sqrt(dd/aa),
                          candidate_to_reference_norm_ratio=math.sqrt(bb/aa),
                          reference_l2=math.sqrt(aa), candidate_l2=math.sqrt(bb))
    return result
