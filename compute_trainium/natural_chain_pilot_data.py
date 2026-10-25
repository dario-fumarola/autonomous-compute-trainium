"""Frozen source-window/task routing for the bounded continuation pilot."""
import math
import torch
from .natural_chain_mixed import visible_stage


def exposure_state(config,exposure,seed,positions):
    if min(exposure,seed)<0 or positions not in [128,1024]:raise ValueError('Declared exposure/noise/context required')
    generator=torch.Generator(device='cpu').manual_seed((seed+exposure)%(2**63-1));std=math.sqrt(2/(5*config.dim))
    value=torch.empty(1,positions,config.dim,dtype=torch.float32)
    return torch.nn.init.trunc_normal_(value,std=std,a=-3*std,b=3*std,generator=generator)


def source_window(packed,window,config,noise_seed):
    batch=packed.batch([window],0)[0]
    return batch.inputs,batch.targets,batch.target_mask,torch.zeros_like(batch.target_mask),exposure_state(config,window,noise_seed,1024)


def task_row(tasks,update,slot,case,config,noise_seed):
    stage=visible_stage(update,case['total_updates']);problem=int(tasks['order'][update,slot]);index=int(tasks['view_indices'][stage,problem])
    arrays=[tasks[k][index:index+1] for k in ['inputs','targets','answer_mask','termination_mask']]
    state=exposure_state(config,10**9+update*128+slot,noise_seed,128)
    return (*arrays,state),index


def answer_count(tasks,update,case):
    stage=visible_stage(update,case['total_updates']);problems=tasks['order'][update];indices=tasks['view_indices'][stage,problems]
    return int(tasks['answer_mask'][indices].sum())
