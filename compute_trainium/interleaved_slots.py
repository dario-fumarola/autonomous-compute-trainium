"""Equivalent interleaving with explicit slot-axis unbinding.

This changes graph layout only. It does not change latent states, depth,
positions, parameters, objective, precision, or nominal matrix work.
"""
import torch
from torch.nn import functional as F
from .interleaved_workspace import InterleavedWorkspace


class SlotInterleavedWorkspace(InterleavedWorkspace):
    @staticmethod
    def source_and_latent(expanded):
        b, length, width = expanded.shape
        if length % 2:
            raise ValueError('Expanded history must contain complete source/latent pairs')
        return expanded.reshape(b, length//2, 2, width).unbind(2)

    def jacobi(self, tokens, refinements=2, *, mode='latent', return_states=False):
        if type(refinements) is not int or refinements<0:
            raise ValueError('Nonnegative integer refinements required')
        if mode not in ('ordinary','latent','pause'):
            raise ValueError('Unknown workspace mode')
        source=self.embeddings(tokens); pos=torch.arange(source.shape[1],device=tokens.device)
        if mode=='ordinary':
            return F.linear(self.body_at(source,pos),self.prediction_weight)
        if mode=='pause':
            latent=self.source_scale*self.embedding.weight.mean(0).expand_as(source)
        else:
            latent=self.body_at(source,pos)
            for _ in range(refinements):
                latent,_=self.source_and_latent(self.body_at(self.interleave(source,latent),pos.repeat_interleave(2)))
        _,final=self.source_and_latent(self.body_at(self.interleave(source,latent),pos.repeat_interleave(2)))
        logits=F.linear(final,self.prediction_weight)
        return (logits,latent) if return_states else logits
