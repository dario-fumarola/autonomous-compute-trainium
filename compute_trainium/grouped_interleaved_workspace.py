"""Equivalent grouped storage for chronological source/latent histories.

Storage order changes, causal access does not. This avoids the measured Native
stack/interleave-to-body execution cliff; no state or gradient is detached.
"""
import torch
from torch.nn import functional as F
from .interleaved_workspace import InterleavedWorkspace
from .probability_workspace import MathBlock


class GroupedInterleavedWorkspace(InterleavedWorkspace):
    def __init__(self,config,*,explicit_attention=False,scale_embeddings=True):
        super().__init__(config,explicit_attention=explicit_attention,scale_embeddings=scale_embeddings)
        self.register_buffer('grouped_cos',torch.cat((self.cos,self.cos),2),persistent=False)
        self.register_buffer('grouped_sin',torch.cat((self.sin,self.sin),2),persistent=False)
        order=torch.cat((2*torch.arange(config.max_seq),2*torch.arange(config.max_seq)+1))
        self.register_buffer('grouped_allowed',order[None,:]<=order[:,None],persistent=False)

    def grouped_body(self,source,latent):
        n=source.shape[1]
        if latent.shape!=source.shape or n>self.config.max_seq:raise ValueError('Matching bounded source/latent slots required')
        if n==self.config.max_seq:
            cos,sin,allowed=self.grouped_cos,self.grouped_sin,self.grouped_allowed
        else:
            cos=torch.cat((self.cos[:,:,:n],self.cos[:,:,:n]),2)
            sin=torch.cat((self.sin[:,:,:n],self.sin[:,:,:n]),2)
            order=torch.cat((2*torch.arange(n,device=source.device),2*torch.arange(n,device=source.device)+1))
            allowed=order[None,:]<=order[:,None]
        h=torch.cat((source,latent),1)
        for block in self.blocks:h=block(h,cos.to(h.dtype),sin.to(h.dtype),allowed)
        return MathBlock.rms(h,self.norm) if self.explicit_attention else self.norm(h)

    def jacobi(self,tokens,refinements=2,*,mode='latent',return_states=False):
        if type(refinements) is not int or refinements<0:raise ValueError('Nonnegative integer refinements required')
        if mode not in ('ordinary','latent','pause'):raise ValueError('Unknown workspace mode')
        source=self.embeddings(tokens);n=source.shape[1];pos=torch.arange(n,device=tokens.device)
        if mode=='ordinary':return F.linear(self.body_at(source,pos),self.prediction_weight)
        if mode=='pause':latent=self.source_scale*self.embedding.weight.mean(0).expand_as(source)
        else:
            latent=self.body_at(source,pos)
            for _ in range(refinements):latent=self.grouped_body(source,latent)[:,:n]
        final=self.grouped_body(source,latent)[:,n:]
        logits=F.linear(final,self.prediction_weight)
        return (logits,latent) if return_states else logits
