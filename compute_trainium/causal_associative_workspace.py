"""Observed-prefix associative memory;current/future outcomes never enter reads."""
import torch
from torch import nn
from torch.nn import functional as F


def prefix_weights(scores,anchors):
    allowed=torch.arange(scores.shape[-1],device=scores.device)[None,:]<anchors[:,None]
    safe=scores.double().masked_fill(~allowed,float('-inf'))
    safe=torch.where((anchors>0)[None,:,None],safe,torch.zeros_like(safe))
    return safe.softmax(-1)*allowed


def target_mass(weights,observed_outcomes,anchors,targets,*,rotate=False):
    values=observed_outcomes[:,None,:].expand(-1,len(anchors),-1)
    if rotate:
        index=(torch.arange(values.shape[-1],device=values.device)[None,:]-1).remainder(anchors.clamp_min(1)[:,None]);values=observed_outcomes[:,None,:].expand_as(values).gather(-1,index[None,:,:].expand(values.shape[0],-1,-1))
    return (weights*(values==targets[:,:,None])).sum(-1)


def mixture_log_probability(raw_logp,mass,gate):
    # Double precision target probabilities avoid log(0)'s undefined derivative
    # at the exactly ordinary initial mixture. The base is frozen.
    probability=(1-gate.double())*raw_logp.double().exp()+gate.double()*mass.double()
    result=probability.clamp_min(1e-300).log()
    # Remove exp/log roundoff at the exact ordinary boundary while retaining
    # the analytic mixture derivative,including the strength's first update.
    identity=torch.where(gate==0,raw_logp.double(),result)
    return result+(identity-result).detach()


class AssociativeWorkspace(nn.Module):
    def __init__(self,hidden=2048,width=128,*,mode='learned'):
        super().__init__()
        if mode not in ['bag','cosine','learned']:raise ValueError('Registered memory reader required')
        self.mode=mode;self.gate=nn.Linear(hidden,1);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias);self.strength=nn.Parameter(torch.zeros(()))
        if mode!='bag':self.log_temperature=nn.Parameter(torch.tensor(16.).log())
        if mode=='learned':
            self.query=nn.Linear(hidden,width,bias=False);self.key=nn.Linear(hidden,width,bias=False)
            with torch.no_grad():self.key.weight.copy_(self.query.weight)

    def read(self,features,anchors):
        current=features.index_select(1,anchors);normalized=F.rms_norm(current,(current.shape[-1],))
        gate=.5*self.strength.clamp(0,1)*self.gate(normalized).squeeze(-1).sigmoid()*(anchors>0)
        if self.mode=='bag':scores=torch.zeros(features.shape[0],len(anchors),features.shape[1],device=features.device,dtype=features.dtype)
        else:
            q=self.query(current) if self.mode=='learned' else current;k=self.key(features) if self.mode=='learned' else features
            q=F.normalize(q,dim=-1);k=F.normalize(k,dim=-1);scores=(q@k.transpose(-1,-2))*self.log_temperature.clamp(0,torch.tensor(256.).log().item()).exp()
        return prefix_weights(scores,anchors),gate

    def forward(self,features,observed_outcomes,anchors,targets,raw_logp,*,rotate=False):
        weights,gate=self.read(features,anchors);mass=target_mass(weights,observed_outcomes,anchors,targets,rotate=rotate)
        return mixture_log_probability(raw_logp,mass,gate)

    @torch.no_grad()
    def project_(self):
        # Projected AdamW avoids getting stuck below the zero-workspace boundary.
        self.strength.clamp_(0,1)
        if self.mode!='bag':self.log_temperature.clamp_(0,torch.tensor(256.).log().item())


def fixed_cache(features,outcomes,anchors,targets,raw_logp,temperature,weight):
    q=F.normalize(features.index_select(1,anchors),dim=-1);k=F.normalize(features,dim=-1);weights=prefix_weights((q@k.transpose(-1,-2))*temperature,anchors);mass=target_mass(weights,outcomes,anchors,targets);gate=torch.full_like(mass,weight)*(anchors>0)
    return mixture_log_probability(raw_logp,mass,gate)


def sequential_reference(model,features,outcomes,anchors,targets,raw_logp):
    """Independent prefix truncation,without masked full-sequence reads."""
    rows=[]
    for qi,t in enumerate(anchors.tolist()):
        if t==0:rows.append(raw_logp[:,qi].double());continue
        current=features[:,t];past=features[:,:t];gate=.5*model.strength.clamp(0,1)*model.gate(F.rms_norm(current,(current.shape[-1],))).squeeze(-1).sigmoid()
        if model.mode=='bag':weights=torch.ones(features.shape[0],t,dtype=features.dtype,device=features.device)/t
        else:
            q=model.query(current) if model.mode=='learned' else current;k=model.key(past) if model.mode=='learned' else past
            scores=(F.normalize(q,dim=-1)[:,None,:]*F.normalize(k,dim=-1)).sum(-1)*model.log_temperature.clamp(0,torch.tensor(256.).log().item()).exp();weights=scores.double().softmax(-1)
        mass=(weights*(outcomes[:,:t]==targets[:,qi,None])).sum(-1);rows.append(mixture_log_probability(raw_logp[:,qi],mass,gate))
    return torch.stack(rows,1)
