"""Compile existing pass boundaries without detaching intermediate states."""
import torch
from torch import nn
from torch.nn import functional as F
from .interleaved_objective import target_loss_sum
from .stable_target_loss import stable_target_log_probs


class InitialPass(nn.Module):
    def __init__(self, model):super().__init__();self.model=model
    def forward(self,x):
        source=self.model.embeddings(x)
        pos=torch.arange(x.shape[1],device=x.device)
        return source,self.model.body_at(source,pos)


class RefinePass(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,source,latent):
        pos=torch.arange(source.shape[1],device=source.device).repeat_interleave(2)
        return self.model.body_at(self.model.interleave(source,latent),pos)[:,::2]


class ReadPass(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,source,latent,y,mask):
        pos=torch.arange(source.shape[1],device=source.device).repeat_interleave(2)
        hidden=self.model.body_at(self.model.interleave(source,latent),pos)[:,1::2]
        logits=F.linear(hidden,self.model.prediction_weight)
        return -(stable_target_log_probs(logits,y)*mask).sum()


class WholeObjective(nn.Module):
    def __init__(self,model,refinements,mode):
        super().__init__();self.model=model;self.refinements=refinements;self.mode=mode
    def forward(self,x,y,mask):
        return target_loss_sum(self.model,x,y,mask,self.refinements,self.mode)


class SegmentedObjective:
    """Not itself compiled: autograd connects the three compiled modules."""
    def __init__(self,model,refinements=2,*,mode='latent',compile_module=lambda m:m):
        if type(refinements) is not int or refinements<0 or mode not in ('latent','ordinary','pause'):
            raise ValueError('Valid registered computation specification required')
        self.refinements=refinements;self.mode=mode
        if mode=='latent':
            self.initial=compile_module(InitialPass(model))
            self.refine=compile_module(RefinePass(model))
            self.read=compile_module(ReadPass(model))
        else:self.whole=compile_module(WholeObjective(model,refinements,mode))

    def __call__(self,x,y,mask):
        if self.mode!='latent':return self.whole(x,y,mask)
        source,latent=self.initial(x)
        for _ in range(self.refinements):latent=self.refine(source,latent)
        return self.read(source,latent,y,mask)
