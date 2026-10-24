"""Same precise Muon arithmetic with runtime scalar transfers, no fill JIT."""
import torch

from .parcae_optimizer_candidate import MuonMatrixOptimizer
from .parcae_optimizer_precise import ParcaePreciseOptimizer


class RuntimeScalarMuon(MuonMatrixOptimizer):
    @torch.no_grad()
    def step(self):
        groups=self.param_groups;first=groups[0]['params'][0]
        # A value-changing static fill dispatch recompiles on the native beta.
        # One host->device tensor transfer retains identical FP32 scalar values.
        rows=[]
        for group in groups:
            shape=tuple(group['params'][0].shape)
            rows.append([group['momentum'],group['lr']*max(1.,shape[-2]/shape[-1])**.5,
                         group['weight_decay'],group.get('beta2',.95)])
        table=torch.tensor(rows,dtype=torch.float32,device='cpu').to(first.device)
        for group,values in zip(groups,table.unbind(0),strict=True):
            parameters=group['params'];first=parameters[0];shape=tuple(first.shape)
            if any(p.grad is None for p in parameters):raise ValueError('Missing Muon gradient')
            reduction=-1 if shape[-2]>=shape[-1] else -2;state=self.state[first]
            if not state:
                state['m']=torch.zeros((len(parameters),*shape),device=first.device,dtype=torch.float32)
                vshape=(len(parameters),shape[-2],1) if reduction==-1 else (len(parameters),1,shape[-1])
                state['v']=torch.zeros(vshape,device=first.device,dtype=torch.float32)
            momentum,lr,decay,beta2=values.unbind(0)
            packed=torch.stack(parameters)
            self.tensor_step(torch.stack([p.grad for p in parameters]),packed,state['m'],state['v'],
                momentum,lr,decay,beta2,group['ns_steps'],reduction)
            for parameter,updated in zip(parameters,packed.unbind(0),strict=True):parameter.copy_(updated)


class ParcaeRuntimeScalarOptimizer(ParcaePreciseOptimizer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Same parameter groups, mathematical recipe, moment/state layout and
        # tensor-step function. Only non-historical scalar ingestion changes.
        self.muon.__class__=RuntimeScalarMuon


def changing_qualification_schedule(optimizer,update):
    import math
    if update<20:
        rate=1e-4*(update+1)/20;momentum=.85+.1*(update+1)/20;decay=.2
    else:
        progress=(update-20)/101
        rate=1e-4*(.55+.45*math.cos(math.pi*progress));momentum=.95;decay=.2*(1-.8*progress)
    for group in optimizer.param_groups:
        group['lr']=rate*group.get('base_lr',1.)
        if group['kind']=='muon':group['momentum']=momentum;group['weight_decay']=decay
    return rate
