"""Mixed natural NTP/original-conclusion pilot primitives;no launch policy."""
import math
import torch
from torch import nn
from .natural_chain_native_loss import answer_parts
from .parcae_candidate import matmul_ledger


class MixedConclusionObjective(nn.Module):
    def __init__(self,model,*,backend=None):
        super().__init__();self.model=model
        self.loss=torch.compile(answer_parts,backend=backend,fullgraph=True,dynamic=False) if backend else answer_parts
    def forward(self,inputs,targets,mask,stop,state,coefficients,n):
        logits=self.model(inputs,state,total_steps=n,backward_cap=n);parts=self.loss(logits,targets,mask,stop)
        return (parts*coefficients).sum(),parts.detach()


def visible_stage(update,total_updates):
    if type(update) is not int or not 0<=update<total_updates or total_updates<4:raise ValueError('Bounded update/endpoint required')
    return 0 if update<total_updates//4 else 1 if update<total_updates//2 else 2


def optimizer_schedule(optimizer,update,total_updates,peak=.001):
    if not 0<=update<total_updates or total_updates<20:raise ValueError('Planned adaptation endpoint required')
    warm=min((update+1)/10,1.);start=total_updates//2
    progress=max(0.,(update+1-start)/(total_updates-start))
    rate=peak*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
    for group in optimizer.param_groups:
        group['lr']=rate*group.get('base_lr',1.)
        if group['kind']=='muon':group['momentum']=.85+.1*warm;group['weight_decay']=.2*(1-.8*progress)
    return rate


def update_matrix_work(config,depth,*,task_weight,optimizer_per_replica):
    ntp=128*matmul_ledger(config,1024,total_steps=depth,backward_cap=depth)['training_matmul_flops']
    task=128*matmul_ledger(config,128,total_steps=depth,backward_cap=depth)['training_matmul_flops'] if task_weight else 0
    return ntp+task+32*optimizer_per_replica


def coefficients(ntp_count,answer_count,*,kind,task_weight,world=32):
    if ntp_count<=0 or answer_count<=0 or not 0<=task_weight<=1:raise ValueError('Whole-update counts and fixed weight required')
    if kind=='ntp':return torch.tensor([world/ntp_count,0.],dtype=torch.float32)
    if kind=='task':return torch.tensor([world*task_weight/answer_count,world*.1*task_weight/128],dtype=torch.float32)
    raise ValueError('Only original NTP or ordered source-answer branch allowed')
