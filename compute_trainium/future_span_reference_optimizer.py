"""Named-state adapter to the independent released optimizer,CPU audit only."""
from contextlib import contextmanager

import torch

from .parcae_optimizer_alignment import align_reference_groups
from .parcae_optimizer_precise import independent_fp32_reference
from .parcae_recovery_check import moments
from third_party.parcae_reference import muon_adamw as reference


def reference_optimizer(model,target_names,native_state=None):
    groups=reference.get_muon_param_groups(model.named_parameters(),adamw_lr=1e-4,
        adamw_betas=(.8,.95),adamw_eps=1e-10,adamw_wd=0.,muon_lr=1e-4)
    groups=align_reference_groups(model,groups,target_names)
    optimizer=reference.MuonAdamW(groups)
    if native_state is not None:
        if native_state['recipe']['groups']!=target_names:raise ValueError('Historical named moment grouping differs')
        by_name=moments(native_state);lookup={id(p):n for n,p in model.named_parameters()}
        for group in optimizer.param_groups:
            params=group['params']
            if group['kind']=='adamw':
                for p in params:
                    old=by_name[lookup[id(p)]]
                    optimizer.state[p].update(step=old['step'],m=old['m'].clone(),v=old['v'].clone())
            else:
                optimizer.state[params[0]].update(
                    m=torch.stack([by_name[lookup[id(p)]]['m'].clone() for p in params]),
                    v=torch.stack([by_name[lookup[id(p)]]['v'].clone() for p in params]))
    return optimizer


@contextmanager
def fp32_reference_execution():
    adam,muon=reference.adamw_step,reference.muon_step
    precise=independent_fp32_reference(reference)
    reference.adamw_step=getattr(adam,'_torchdynamo_orig_callable',adam)
    reference.muon_step=precise
    try:yield
    finally:reference.adamw_step,reference.muon_step=adam,muon
