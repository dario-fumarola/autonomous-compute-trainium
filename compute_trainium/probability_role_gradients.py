"""Identical forward map with separate leaf tensors for embedding-role diagnosis."""
import torch.nn.functional as F
from .probability_workspace import expected_embedding

def role_replay(model,tokens,input_weight,output_weight,proposal_weight,extra_steps):
    state=F.embedding(tokens,input_weight)
    for _ in range(extra_steps):
        logits=F.linear(model.body(state),output_weight)
        state=state+expected_embedding(logits,proposal_weight)
    return F.linear(model.body(state),output_weight)
