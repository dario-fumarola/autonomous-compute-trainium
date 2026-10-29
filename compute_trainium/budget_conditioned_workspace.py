"""Tell proposal passes how many replay steps remain; final readout is unshifted."""
import torch
from torch import nn
from torch.nn import functional as F
from .probability_workspace import ProbabilityWorkspace,expected_embedding

class BudgetConditionedWorkspace(ProbabilityWorkspace):
    def __init__(self,config,*,explicit_attention=False,maximum_extra_steps=2):
        super().__init__(config,explicit_attention=explicit_attention)
        if type(maximum_extra_steps) is not int or maximum_extra_steps<1:raise ValueError('Positive supported cue count required')
        self.maximum_extra_steps=maximum_extra_steps
        # Zero initialization preserves every original initial forward and RNG.
        self.remaining_budget_cues=nn.Parameter(torch.zeros(maximum_extra_steps,config.dim))

    def forward(self,tokens,extra_steps=0,proposal='expected'):
        return self.replay_with_permutation(tokens,extra_steps,proposal,None)

    def replay_with_permutation(self,tokens,extra_steps,proposal,permutation):
        if type(extra_steps) is not int or not 0<=extra_steps<=self.maximum_extra_steps:raise ValueError('Unsupported replay budget')
        if proposal not in ('expected','uniform'):raise ValueError('Unknown proposal')
        if permutation is not None and proposal!='expected':raise ValueError('Only generated proposals can be permuted')
        weight=self.embedding.weight;value_weight=weight if permutation is None else weight.index_select(0,permutation)
        state=self.embedding(tokens)
        for remaining in range(extra_steps,0,-1):
            if proposal=='expected':
                logits=F.linear(self.body(state+self.remaining_budget_cues[remaining-1]),weight)
                thought=expected_embedding(logits,value_weight)
            else:thought=weight.mean(0)
            state=state+thought.to(state.dtype)
        logits=F.linear(self.body(state),weight)
        # Connect otherwise-unused cue rows to an exact zero gradient at depth0.
        # This does not change predictions or add an auxiliary objective.
        return logits+self.remaining_budget_cues.sum()*0
