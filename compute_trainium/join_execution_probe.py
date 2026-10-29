"""Equivalent persistent-slot arrangements; numerical fixtures only."""
import torch
from .body_execution_probe import Fixture as BodyFixture
from .probability_workspace import MathBlock

CASES=('stack','feature_cat','sequence_gather','repeat_where','batch_permute','stack_clone','grouped_history','stack_zero')

class Fixture(BodyFixture):
    def __init__(self,case,reference=False,small=False):
        super().__init__(4,reference,small);self.case=case
        indices=torch.arange(self.n//2)
        order=torch.stack((indices,indices+self.n//2),-1).flatten()
        self.register_buffer('order',order)
        inverse=torch.argsort(order);self.register_buffer('inverse',inverse)
    def input(self):
        return torch.randn(2,self.n,self.d,generator=torch.Generator().manual_seed(9872))
    def forward(self,x):
        n=self.n//2;s=x[:,:n];z=x[:,n:];c=self.case
        if self.reference:c=0
        if c==0:joined=torch.stack((s,z),2).flatten(1,2)
        elif c==1:joined=torch.cat((s,z),-1).reshape(x.shape)
        elif c==2:joined=torch.cat((s,z),1).index_select(1,self.order)
        elif c==3:
            odd=(torch.arange(self.n,device=x.device)%2).bool()[None,:,None]
            joined=torch.where(odd,z.repeat_interleave(2,1),s.repeat_interleave(2,1))
        elif c==4:joined=torch.cat((s,z),0).reshape(2,2,n,self.d).permute(1,2,0,3).reshape(x.shape)
        elif c==5:joined=torch.stack((s,z),2).flatten(1,2).clone(memory_format=torch.contiguous_format)
        elif c==6:
            joined=torch.cat((s,z),1)
            cos=self.cos.index_select(2,self.inverse);sin=self.sin.index_select(2,self.inverse)
            chronology=self.inverse
            allowed=chronology[:,None]>=chronology[None,:]
            for block in self.blocks:joined=block(joined,cos,sin,allowed)
            out=self.norm(joined) if self.reference else MathBlock.rms(joined,self.norm)
            return out[:,:n]
        else:joined=torch.stack((s,z),2).flatten(1,2)+torch.zeros_like(x)
        return self.body(joined)[:,::2]
