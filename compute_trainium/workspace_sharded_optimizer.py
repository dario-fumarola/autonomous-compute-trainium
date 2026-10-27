"""Local FP32 AdamW and EMA over the existing full-parameter DTensor shards."""
import torch
from torch import nn
import torch.distributed as dist
from .scale_sharded_candidate import local_parameter


class WorkspaceShardedAdamW:
    def __init__(self,model,*,lr=1e-5,fused=False):
        self.named=list(model.named_parameters());self.world=dist.get_world_size()
        self.local=[nn.Parameter(local_parameter(p).detach()) for _,p in self.named]
        self.adam=torch.optim.AdamW(self.local,lr=lr,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False,fused=fused)
        self.update=0
    def zero_grad(self):
        for _,p in self.named:p.grad=None
        self.adam.zero_grad(set_to_none=True)
    @torch.no_grad()
    def finish_gradients(self,*,scale=1.,clip=1.):
        if scale<=0 or clip<=0:raise ValueError('Positive gradient scale and clip required')
        total=torch.zeros((),device=self.local[0].device,dtype=torch.float32)
        for _,p in self.named:
            if p.grad is None:raise ValueError('Every parameter needs a gradient')
            p.grad=p.grad.redistribute(placements=p.placements);g=p.grad.to_local();g.div_(scale)
            total.add_(g.square().sum()*(1/self.world if p.placements[0].is_replicate() else 1.))
        dist.all_reduce(total);norm=total.sqrt();value=float(norm.cpu())
        if not torch.isfinite(torch.tensor(value)):raise FloatingPointError('Nonfinite global gradient')
        factor=(clip/(norm+1e-6)).clamp(max=1.)
        for _,p in self.named:p.grad.to_local().mul_(factor)
        return value
    def step(self):
        for local,(_,p) in zip(self.local,self.named,strict=True):
            if p.grad is None:raise ValueError('All gradients required before an optimizer update')
            local.grad=p.grad.to_local()
        self.adam.step();self.update+=1
    def state_dict(self):return dict(update=self.update,names=[n for n,_ in self.named],adam=self.adam.state_dict())
    def load_state_dict(self,value):
        if value['names']!=[n for n,_ in self.named]:raise ValueError('Exact parameter order required')
        self.adam.load_state_dict(value['adam']);self.update=value['update']


@torch.no_grad()
def sharded_ema(scorer,actor,decay=.99):
    left=list(scorer.named_parameters());right=list(actor.named_parameters())
    if not 0<=decay<1 or [n for n,_ in left]!=[n for n,_ in right]:raise ValueError('Matching models and valid EMA decay required')
    for (_,target),(_,source) in zip(left,right,strict=True):
        a,b=local_parameter(target),local_parameter(source)
        if a.data_ptr()==b.data_ptr():raise ValueError('Independent actor/scorer shards required')
        a.mul_(decay).add_(b,alpha=1-decay)
