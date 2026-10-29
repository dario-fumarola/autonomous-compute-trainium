"""Persistent continuous tokens and finite Jacobi training.

Inspired by PonderLM-2, not a paper reproduction. One latent token per source
position, shared source/latent rotary positions, complete gradients, final NTP.
"""
import math
import torch
from torch.nn import functional as F
from .untied_probability_workspace import UntiedProbabilityWorkspace
from .probability_workspace import MathBlock


class InterleavedWorkspace(UntiedProbabilityWorkspace):
    def __init__(self, config, *, explicit_attention=False, scale_embeddings=True):
        super().__init__(config, explicit_attention=explicit_attention)
        self.source_scale = math.sqrt(2.5*config.dim) if scale_embeddings else 1.

    def body_at(self, embedding, positions):
        h=embedding
        cos=self.cos.index_select(2,positions).to(h.dtype)
        sin=self.sin.index_select(2,positions).to(h.dtype)
        for block in self.blocks:h=block(h,cos,sin)
        return MathBlock.rms(h,self.norm) if self.explicit_attention else self.norm(h)

    def embeddings(self,tokens):
        if tokens.shape[1]>self.config.max_seq:raise ValueError('Source context exceeds rotary cache')
        return self.source_scale*self.embedding(tokens)

    @staticmethod
    def interleave(source,latent):
        return torch.stack((source,latent),dim=2).flatten(1,2)

    def jacobi(self,tokens,refinements=2,*,mode='latent',return_states=False):
        if type(refinements) is not int or refinements<0:raise ValueError('Nonnegative integer refinements required')
        if mode not in ('ordinary','latent','pause'):raise ValueError('Unknown workspace mode')
        source=self.embeddings(tokens);n=source.shape[1]
        pos=torch.arange(n,device=tokens.device)
        if mode=='ordinary':return F.linear(self.body_at(source,pos),self.prediction_weight)
        if mode=='pause':latent=self.source_scale*self.embedding.weight.mean(0).expand_as(source)
        else:
            latent=self.body_at(source,pos)
            for _ in range(refinements):
                latent=self.body_at(self.interleave(source,latent),pos.repeat_interleave(2))[:,::2]
        final=self.body_at(self.interleave(source,latent),pos.repeat_interleave(2))[:,1::2]
        logits=F.linear(final,self.prediction_weight)
        return (logits,latent) if return_states else logits

    def forward(self,tokens,refinements=2,mode="latent"):
        return self.jacobi(tokens,refinements,mode=mode)

    def sequential(self,tokens,*,return_states=False):
        """Expensive exact reference. No future token is present in any prefix."""
        source=self.embeddings(tokens);history=[];positions=[];states=[];scores=[]
        for t in range(source.shape[1]):
            history.append(source[:,t:t+1]);positions.append(t)
            z=self.body_at(torch.cat(history,dim=1),torch.tensor(positions,device=tokens.device))[:,-1:]
            states.append(z);history.append(z);positions.append(t)
            result=self.body_at(torch.cat(history,dim=1),torch.tensor(positions,device=tokens.device))[:,-1:]
            scores.append(F.linear(result,self.prediction_weight))
        logits=torch.cat(scores,dim=1)
        return (logits,torch.cat(states,dim=1)) if return_states else logits


def matrix_work(config,length,refinements,mode='latent',batch=1):
    """Dense dispatched matmuls; full backward, no hidden no-grad or caching."""
    if type(refinements) is not int or refinements<0 or mode not in ('ordinary','latent','pause'):
        raise ValueError('Invalid workspace specification')
    def body(n):return config.flops(0,n,batch)['forward_matmul_flops']
    head=2*batch*length*config.dim*config.vocab_size
    forward=(body(length) if mode=='ordinary' else body(2*length)) + head
    if mode=='latent':forward+=body(length)+refinements*body(2*length)
    return dict(forward_matmul_flops=forward,training_matmul_flops=3*forward,
                source_body_passes=1 if mode in ('ordinary','latent') else 0,
                expanded_body_passes=0 if mode=='ordinary' else 1+(refinements if mode=='latent' else 0))
