"""Equal-body-pass diagnostic: change proposal content, recompute all states."""
from torch import nn
from torch.nn import functional as F


class ProposalReadout(nn.Module):
    def __init__(self, model, steps, proposal='expected', permutation=None):
        super().__init__();self.model=model;self.steps=steps;self.proposal=proposal
        self.register_buffer('permutation',permutation)

    def forward(self, tokens):
        if hasattr(self.model,'replay_with_permutation'):
            return self.model.replay_with_permutation(tokens,self.steps,self.proposal,self.permutation)
        if self.permutation is None:return self.model(tokens,self.steps,self.proposal)
        if self.proposal!='expected':raise ValueError('Only generated proposals are permuted')
        weight=self.model.embedding.weight;state=self.model.embedding(tokens)
        changed=weight.index_select(0,self.permutation)
        for _ in range(self.steps):
            distribution=F.linear(self.model.body(state),weight).softmax(-1)
            state=state+distribution@changed
        return F.linear(self.model.body(state),weight)
