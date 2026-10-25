"""Normal causal decoder objective for original ordered conclusions."""
import torch
from torch import nn
from torch.nn import functional as F
from .parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate


def answer_parts(logits,targets,answer_mask,termination_mask):
    values=F.cross_entropy(logits.float().flatten(0,1),targets.flatten(),reduction='none').view_as(targets)
    return torch.stack([(values*answer_mask).sum(),(values*termination_mask).sum()])


class NaturalChainObjective(nn.Module):
    def __init__(self,model,*,backend=None):
        super().__init__();self.model=model
        self.loss=torch.compile(answer_parts,backend=backend,fullgraph=True,dynamic=False) if backend else answer_parts

    def forward(self,inputs,targets,answer_mask,termination_mask,state,coefficients,n):
        logits=self.model(inputs,state,total_steps=n,backward_cap=n)
        parts=self.loss(logits,targets,answer_mask,termination_mask)
        return (parts*coefficients).sum(),parts.detach()


def task_coefficients(answer_tokens,examples,*,world=1,termination_weight=.1):
    if min(answer_tokens,examples,world)<=0 or not 0<=termination_weight<=1:
        raise ValueError('Whole-batch positive counts and bounded EOS weight required')
    return torch.tensor([world/answer_tokens,world*termination_weight/examples],dtype=torch.float32)


def task_depth(arm,update):
    if arm=='one':return 1
    if arm=='recurrent':return (4,1,2)[update%3]
    raise ValueError('Explicit one-pass or supported-depth candidate required')


def bridge_visible(steps,update):
    if not 0<=update<=120:raise ValueError('Finite qualification curriculum only')
    return steps if update<40 else steps//2 if update<80 else 0
