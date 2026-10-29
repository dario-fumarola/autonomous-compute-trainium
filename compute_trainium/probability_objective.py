"""Uniform real-token supervision at registered probability-replay exits.

Final-only remains the exact old path. Joint exits change the learning objective,
not source targets, token weights, parameters, or proposal gradients.
"""
import torch.nn.functional as F
from .probability_workspace import expected_embedding
from .stable_target_loss import stable_target_log_probs


def exit_logits(model,tokens,depth,proposal,objective):
    if objective=='final_only':
        return (model(tokens,depth,proposal),)
    if objective!='uniform_exits':raise ValueError('Unknown probability objective')
    if type(depth) is not int or depth not in (0,1) or proposal!='expected':
        raise ValueError('Joint-exit qualification supports only zero/one expected replay')
    if not hasattr(model,'prediction_weight'):
        raise ValueError('Joint-exit qualification requires independent input/output tables')
    state=model.embedding(tokens)
    first=F.linear(model.body(state),model.prediction_weight)
    if depth==0:return (first,)
    thought=expected_embedding(first,model.embedding.weight)
    final=F.linear(model.body(state+thought.to(state.dtype)),model.prediction_weight)
    return first,final


def target_loss_sum(model,tokens,targets,mask,depth,proposal='expected',objective='final_only'):
    logits=exit_logits(model,tokens,depth,proposal,objective)
    # Weights sum to one at EVERY real target. No selected-position emphasis.
    total=-(stable_target_log_probs(logits[0],targets)*mask).sum()
    for other in logits[1:]:total=total-(stable_target_log_probs(other,targets)*mask).sum()
    return total/len(logits) if len(logits)>1 else total


def reference_loss_sum(model,tokens,targets,mask,depth,proposal='expected',objective='final_only'):
    """Independent CE and replay expression for CPU audits, with SDPA model."""
    def ce(logits):
        return (F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view_as(targets)*mask).sum()
    if objective=='final_only':return ce(model(tokens,depth,proposal))
    if objective!='uniform_exits' or type(depth) is not int or depth not in (0,1) or proposal!='expected' or not hasattr(model,'prediction_weight'):
        raise ValueError('Unqualified objective specification')
    embeddings=model.embedding(tokens)
    shallow=model.body(embeddings)@model.prediction_weight.T
    if depth==0:return ce(shallow)
    refined=model.body(embeddings+shallow.softmax(-1)@model.embedding.weight)@model.prediction_weight.T
    return (ce(shallow)+ce(refined))*.5
