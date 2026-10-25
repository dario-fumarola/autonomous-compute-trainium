"""Local CPU-math-compatible attention candidate;never patches the SDK.

The installed Math2 implementation rounds the saved attention output before
using it in a backward correction. Upcasting the attention call preserves its
internal output until backward,then casts the public result to the original
dtype. Native operator/global-gradient/cost qualification is still required.
"""
import torch
from torch.nn import functional as F

from .parcae_candidate import Attention
from .parcae_fp16_candidate import ParcaeFP16Candidate


def math_sdpa(query,key,value,*,is_causal=True):
    dtype=query.dtype
    with torch.autocast(query.device.type,enabled=False):
        return F.scaled_dot_product_attention(query.float(),key.float(),value.float(),
            is_causal=is_causal).to(dtype)


class MathSDPAAttention(Attention):
    def forward(self,x,freqs,ve=None):
        b,n,width=x.shape;shape=(b,n,self.heads,self.head_dim)
        q,k,v=self.c_q(x).view(shape),self.c_k(x).view(shape),self.c_v(x).view(shape)
        if ve is not None:
            gate=2*torch.sigmoid(self.ve_gate(x[...,:32]));v=v+gate[...,None]*ve.view(shape)
        pair=torch.cat([q,k],dim=2).unflatten(-1,(-1,2)).float()
        rot=torch.stack([pair[...,0]*freqs[...,0]-pair[...,1]*freqs[...,1],
            pair[...,1]*freqs[...,0]+pair[...,0]*freqs[...,1]],-1).flatten(3)
        q,k=torch.split(rot.to(q.dtype),self.heads,dim=2)
        q,k=F.rms_norm(q,(self.head_dim,)),F.rms_norm(k,(self.head_dim,))
        y=math_sdpa(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        return self.c_proj(y.contiguous().view(b,n,width))


class MathSDPAMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.recurrent:
            layers=[*self.transformer.prelude,*self.transformer.core_block,*self.transformer.coda]
        else:layers=list(self.transformer.h)
        # Constructors consume no additional ongoing randomness and every
        # original tensor is copied exactly before an optimizer is constructed.
        with torch.random.fork_rng(devices=[]):
            for i,block in enumerate(layers):
                old=block.attn
                replacement=MathSDPAAttention(self.config,self.config.has_value_embedding(i))
                replacement.load_state_dict(old.state_dict(),strict=True);replacement.train(old.training)
                block.attn=replacement


class ParcaeMathSDPACandidate(MathSDPAMixin,ParcaeFP16Candidate):
    pass
