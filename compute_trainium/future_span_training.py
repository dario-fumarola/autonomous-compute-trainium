"""Fixed-shape objective/batch primitives,not a scientific launch entry point."""
import functools
import math

import torch
from torch import nn

from .future_span import future_span_targets
from .future_span_sparse import sparse_targets,sparse_future_loss_sum
from .metrics import real_token_losses
from .parcae_training_candidate import initial_state


def objective_coefficients(real_targets, valid_anchors, weight, world):
    if (any(type(x) is not int for x in [real_targets,valid_anchors,world])
            or real_targets<=0 or valid_anchors<0 or world<1
            or not math.isfinite(weight) or weight<0):
        raise ValueError('Whole-optimizer-batch counts and finite objective weight required')
    # Runtime tensor values avoid specializing a compiled graph on each batch's
    # varying real-token/valid-anchor denominator. Transfer once per update.
    return torch.tensor([world/real_targets,world*weight/max(1,valid_anchors)],dtype=torch.float32)


def loss_parts(ntp_logits, future_logits, targets, target_mask, token_ids, distinct, valid, *, horizon):
    return torch.stack([real_token_losses(ntp_logits,targets,target_mask).sum(),
        sparse_future_loss_sum(future_logits,token_ids,distinct,valid,horizon)])


class FutureSpanObjective(nn.Module):
    def __init__(self,model,*,backend=None):
        super().__init__();self.model=model
        implementation=functools.partial(loss_parts,horizon=model.future_config.horizon)
        self.loss=(torch.compile(implementation,backend=backend,fullgraph=True,dynamic=False)
                   if backend else implementation)

    def forward(self,inputs,targets,target_mask,state,token_ids,distinct,valid,coefficients,n,k):
        ntp_logits,future_logits=self.model(inputs,state,total_steps=n,backward_cap=k)
        parts=self.loss(ntp_logits,future_logits,targets,target_mask,token_ids,distinct,valid)
        return (parts*coefficients).sum(),parts.detach()


def training_window(packed,window,model_config,future_config,state_seed):
    if packed.context!=future_config.context or packed.context>1024:
        raise ValueError('Matching context within the qualified1024-position state geometry required')
    item=packed.batch([window],0)[0] # Existing coverage/boundary policy stays exact.
    start=window*packed.context
    source=packed.tokens[start:start+packed.context+1].long()[None]
    real=packed.domains[start:start+packed.context+1][None]>0
    labels=sparse_targets(future_span_targets(source,real,future_config,model_config.vocab_size))
    return (item.inputs,item.targets,item.target_mask,
            initial_state(model_config,window,state_seed)[:,:packed.context],
            labels.token_ids,labels.distinct,labels.valid)
