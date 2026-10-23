"""Explicit common LM optimizer for the new experiment, not legacy migration."""
import torch


def parameter_groups(model, embedding_parameter, decay=.1):
    matrix,no_decay=[],[]
    for parameter in model.parameters():
        if not parameter.requires_grad:
            raise ValueError('Every model parameter must remain trainable')
        (no_decay if parameter is embedding_parameter or parameter.ndim<2 else matrix).append(parameter)
    ids=[id(p) for group in [matrix,no_decay] for p in group]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters()}:
        raise AssertionError('Optimizer grouping duplicates or omits a parameter')
    if not matrix or not no_decay or id(embedding_parameter) not in ids:
        raise ValueError('Expected matrix weights and a tied embedding parameter')
    return [{'params':matrix,'weight_decay':decay},{'params':no_decay,'weight_decay':0.}]


def make_optimizer(model, lr, *, fused):
    embedding=model.decoder.tok_embeddings.weight
    if embedding is not model.decoder.lm_head.weight:
        raise ValueError('Expected tied embedding and head')
    return torch.optim.AdamW(parameter_groups(model,embedding),lr=lr,betas=(.9,.95),eps=1e-8,
                             weight_decay=0.,foreach=False,fused=fused)
