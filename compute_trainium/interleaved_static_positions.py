"""Equivalent fixed-position RoPE caches, avoiding device index selection.

The original indexed body remains available for exact sequential references.
Only full-prefix ordinary/interleaved executions use these nonpersistent caches.
"""
from torch.nn import functional as F
from .interleaved_workspace import InterleavedWorkspace
from .probability_workspace import MathBlock


class StaticPositionInterleavedWorkspace(InterleavedWorkspace):
    def __init__(self, config, *, explicit_attention=False, scale_embeddings=True):
        super().__init__(config, explicit_attention=explicit_attention,
                         scale_embeddings=scale_embeddings)
        self.register_buffer('expanded_cos',self.cos.repeat_interleave(2,2),persistent=False)
        self.register_buffer('expanded_sin',self.sin.repeat_interleave(2,2),persistent=False)

    def fixed_body(self, embedding, *, expanded):
        n=embedding.shape[1]
        if n>self.config.max_seq*(2 if expanded else 1):
            raise ValueError('Input exceeds fixed rotary cache')
        cos=(self.expanded_cos if expanded else self.cos)[:,:,:n].to(embedding.dtype)
        sin=(self.expanded_sin if expanded else self.sin)[:,:,:n].to(embedding.dtype)
        h=embedding
        for block in self.blocks:h=block(h,cos,sin)
        return MathBlock.rms(h,self.norm) if self.explicit_attention else self.norm(h)

    def jacobi(self,tokens,refinements=2,*,mode='latent',return_states=False):
        if type(refinements) is not int or refinements<0:
            raise ValueError('Nonnegative integer refinements required')
        if mode not in ('ordinary','latent','pause'):
            raise ValueError('Unknown workspace mode')
        source=self.embeddings(tokens)
        if mode=='ordinary':
            return F.linear(self.fixed_body(source,expanded=False),self.prediction_weight)
        if mode=='pause':
            latent=self.source_scale*self.embedding.weight.mean(0).expand_as(source)
        else:
            latent=self.fixed_body(source,expanded=False)
            for _ in range(refinements):
                latent=self.fixed_body(self.interleave(source,latent),expanded=True)[:,::2]
        final=self.fixed_body(self.interleave(source,latent),expanded=True)[:,1::2]
        logits=F.linear(final,self.prediction_weight)
        return (logits,latent) if return_states else logits
