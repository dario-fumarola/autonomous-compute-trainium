"""Independent output rows W and input/workspace-value rows E."""
import torch
from torch import nn
from torch.nn import functional as F
from .probability_workspace import ProbabilityWorkspace,expected_embedding

class UntiedProbabilityWorkspace(ProbabilityWorkspace):
    def __init__(self,config,*,explicit_attention=False):
        super().__init__(config,explicit_attention=explicit_attention)
        # An identical start isolates untying rather than another initial draw.
        self.prediction_weight=nn.Parameter(self.embedding.weight.detach().clone())

    def additional_no_decay_parameters(self):
        return (self.prediction_weight,)

    def forward(self,tokens,extra_steps=0,proposal='expected'):
        return self.replay_with_permutation(tokens,extra_steps,proposal,None)

    def replay_with_permutation(self,tokens,extra_steps,proposal,permutation):
        if type(extra_steps) is not int or extra_steps<0:raise ValueError('Nonnegative integer replay budget required')
        if proposal not in ('expected','uniform'):raise ValueError('Unknown proposal')
        if permutation is not None and proposal!='expected':raise ValueError('Only generated proposals can be permuted')
        values=self.embedding.weight
        if permutation is not None:values=values.index_select(0,permutation)
        state=self.embedding(tokens)
        for _ in range(extra_steps):
            if proposal=='expected':
                logits=F.linear(self.body(state),self.prediction_weight)
                thought=expected_embedding(logits,values)
            else:thought=self.embedding.weight.mean(0)
            state=state+thought.to(state.dtype)
        return F.linear(self.body(state),self.prediction_weight)
