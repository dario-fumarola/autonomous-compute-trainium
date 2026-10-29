"""CPU-only causal readout interventions; never a fitted architecture.

Recompute every downstream state. Preserve positions, sequence length and the
attention rectangles while blocking access to earlier latent memory. The
current latent slot can still attend itself. Such an intervention measures
dependence on memory, not reasoning or in-distribution deployment quality.
"""
import torch
from torch.nn import functional as F

from .probability_workspace import MathBlock


def memory_mask(is_latent, device=None, *, block_earlier_latents=False):
    n = len(is_latent)
    if not n or any(type(x) is not bool for x in is_latent):
        raise ValueError('Nonempty explicit chronological slot inventory required')
    allowed = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    if block_earlier_latents:
        latent = torch.tensor(is_latent, dtype=torch.bool, device=device)
        allowed &= ~latent[None, :] | torch.eye(n, dtype=torch.bool, device=device)
    return allowed


def readout(model, tokens, stride=128, *, block_earlier_latents=False):
    """Exactly one supported slot per full chunk; no targets or donor selection."""
    if type(stride) is not int or stride < 1 or type(block_earlier_latents) is not bool:
        raise ValueError('Explicit stride and intervention required')
    source = model.embeddings(tokens)
    pieces, positions, latent, selected = [], [], [], []
    stored = 0

    def body():
        h = torch.cat(pieces, 1)
        p = torch.tensor(positions, device=tokens.device)
        cos = model.cos.index_select(2, p).to(h.dtype)
        sin = model.sin.index_select(2, p).to(h.dtype)
        mask = memory_mask(latent, tokens.device,
                           block_earlier_latents=block_earlier_latents)
        for block in model.blocks:
            h = block(h, cos, sin, mask)
        return MathBlock.rms(h, model.norm) if model.explicit_attention else model.norm(h)

    for left in range(0, source.shape[1], stride):
        right = min(left + stride, source.shape[1])
        pieces.append(source[:, left:right])
        positions.extend(range(left, right))
        latent.extend([False] * (right-left))
        selected.extend(range(stored, stored+right-left))
        stored += right-left
        if right % stride == 0:
            proposal = body()[:, -1:]
            pieces.append(proposal)
            positions.append(right-1)
            latent.append(True)
            selected[-1] = stored
            stored += 1
    hidden = body().index_select(1, torch.tensor(selected, device=tokens.device))
    return F.linear(hidden, model.prediction_weight)
