"""Uniform authentic-target objective; no loss on generated latent slots."""
from torch.nn import functional as F
from .model import ModelConfig
from .interleaved_workspace import InterleavedWorkspace
from .stable_target_loss import stable_target_log_probs

def make_model(plan,*,explicit_attention=False):
    return InterleavedWorkspace(ModelConfig(**plan['model_config']),explicit_attention=explicit_attention)

def target_loss_sum(model,x,y,mask,refinements,mode='latent'):
    return -(stable_target_log_probs(model(x,refinements,mode),y)*mask).sum()

def reference_loss_sum(model,x,y,mask,refinements,mode='latent'):
    logits=model(x,refinements,mode)
    return (F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)*mask).sum()
