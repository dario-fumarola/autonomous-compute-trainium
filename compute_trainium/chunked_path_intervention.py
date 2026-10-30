"""Read-only two-factor intervention on sparse workspace paths.

All source and slot states are recomputed. The source-boundary/blocked-memory
combination equals the same weights' ordinary path but still executes slots.
It is an intervention, not a trained deployment mode or a fit prescription.
"""
import torch
from torch.nn import functional as F

from .chunked_memory_intervention import memory_mask
from .probability_workspace import MathBlock


def readout(model, tokens, stride=128, *, mode='latent',
            source_boundary=False, block_memory=False):
    if type(stride) is not int or stride < 1 or mode not in ('latent','pause'):
        raise ValueError('Explicit positive stride and workspace input required')
    if type(source_boundary) is not bool or type(block_memory) is not bool:
        raise ValueError('Explicit independent intervention switches required')
    source = model.embeddings(tokens)
    pieces,positions,latent,selected = [],[],[],[]
    stored = 0

    def body():
        h = torch.cat(pieces,1)
        p = torch.tensor(positions,device=tokens.device)
        cos = model.cos.index_select(2,p).to(h.dtype)
        sin = model.sin.index_select(2,p).to(h.dtype)
        mask = memory_mask(latent,tokens.device,block_earlier_latents=block_memory)
        for block in model.blocks:h=block(h,cos,sin,mask)
        return MathBlock.rms(h,model.norm) if model.explicit_attention else model.norm(h)

    for left in range(0,source.shape[1],stride):
        right = min(left+stride,source.shape[1])
        pieces.append(source[:,left:right]); positions.extend(range(left,right))
        latent.extend([False]*(right-left)); selected.extend(range(stored,stored+right-left))
        stored += right-left
        if right % stride == 0:
            proposal = (body()[:,-1:] if mode=='latent' else
                model.source_scale*model.embedding.weight.mean(0).view(1,1,-1).expand(source.shape[0],1,-1))
            pieces.append(proposal); positions.append(right-1); latent.append(True)
            if not source_boundary:selected[-1]=stored
            stored += 1
    hidden = body().index_select(1,torch.tensor(selected,device=tokens.device))
    return F.linear(hidden,model.prediction_weight)
