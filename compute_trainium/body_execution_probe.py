"""Numerical extraction/whole-body timing; never training examples."""
import torch
from torch import nn
from .model import ModelConfig,Block
from .probability_workspace import MathBlock

CASES=('body6','slice_even','gather_even','contiguous_even','interleaved_slice','interleaved_gather','interleaved_contiguous','two_shared_refines')

class Fixture(nn.Module):
    def __init__(self,case,reference=False,small=False):
        super().__init__();self.case=case;self.reference=reference
        self.d=16 if small else 512;self.n=6 if small else 1024;heads=2 if small else 8
        c=ModelConfig(dim=self.d,layers=2 if small else 6,heads=heads,kv_heads=heads,ffn_dim=32 if small else 1376,vocab_size=29,max_seq=self.n)
        with torch.random.fork_rng():
            torch.manual_seed(9871)
            self.blocks=nn.ModuleList([(Block if reference else MathBlock)(c) for _ in range(c.layers)])
            self.norm=nn.RMSNorm(self.d,eps=1e-5)
        frequency=10000**(-torch.arange(0,self.d//heads,2).float()/(self.d//heads))
        phase=(torch.arange(self.n)//2).float()[:,None]*frequency[None,:]
        self.register_buffer('cos',phase.cos()[None,None]);self.register_buffer('sin',phase.sin()[None,None])
    def input(self):return torch.randn(2,self.n,self.d,generator=torch.Generator().manual_seed(9872))
    def body(self,x):
        for block in self.blocks:x=block(x,self.cos,self.sin)
        return self.norm(x) if self.reference else MathBlock.rms(x,self.norm)
    @staticmethod
    def join(x,z):return torch.stack((x,z),2).flatten(1,2)
    def forward(self,x):
        c=self.case
        if c>=4:
            source=x[:,:self.n//2];latent=x[:,self.n//2:];x=self.join(source,latent)
        out=self.body(x)
        if c==0:return out
        if c in (2,5):return out.index_select(1,torch.arange(0,self.n,2,device=x.device))
        if c in (3,6):return out[:,::2].contiguous()
        if c==7:return self.body(self.join(source,out[:,::2]))[:,::2]
        return out[:,::2]
