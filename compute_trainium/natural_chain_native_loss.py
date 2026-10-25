"""The qualified explicit vocabulary CE for the ordered-conclusion task."""
import torch

def answer_parts(logits,targets,answer_mask,termination_mask):
    values=logits.float()
    terms=values.logsumexp(-1)-values.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    return torch.stack([(terms*answer_mask).sum(),(terms*termination_mask).sum()])
