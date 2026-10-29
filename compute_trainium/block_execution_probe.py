"""Numerical block/operator timing fixtures; no research source examples."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import ModelConfig, Block
from .probability_workspace import MathBlock

CASES=('block512','block1024','rotary512','rotary1024','ffn1024','rms1024','interleave1024','strided_head512')

class Fixture(nn.Module):
    def __init__(self,case,reference=False,small=False):
        super().__init__();self.case=case;self.reference=reference
        d=16 if small else 512;h=2 if small else 8;f=32 if small else 1376
        n=(6 if small else 1024) if '1024' in CASES[case] else (3 if small else 512)
        self.n=n;self.d=d;self.h=h
        c=ModelConfig(dim=d,layers=1,heads=h,kv_heads=h,ffn_dim=f,vocab_size=29 if small else 32000,max_seq=n)
        with torch.random.fork_rng():
            torch.manual_seed(9851+case)
            self.block=(Block if reference else MathBlock)(c)
            self.head=nn.Linear(d,c.vocab_size,bias=False) if case==7 else None
        frequency=10000**(-torch.arange(0,d//h,2).float()/(d//h))
        phase=torch.arange(n).float()[:,None]*frequency[None,:]
        self.register_buffer('cos',phase.cos()[None,None]);self.register_buffer('sin',phase.sin()[None,None])
    def input(self):
        shape=(2,self.n*2,self.d) if self.case==7 else (2,self.n,self.d)
        return torch.randn(shape,generator=torch.Generator().manual_seed(9861+self.case))
    def forward(self,x):
        c=self.case
        if c<2:return self.block(x,self.cos,self.sin)
        if c<4:
            a=x.view(2,self.n,self.h,-1).transpose(1,2)
            even,odd=a[...,::2],a[...,1::2]
            return torch.stack((even*self.cos-odd*self.sin,odd*self.cos+even*self.sin),-1).flatten(-2)
        if c==4:return self.block.down(F.silu(self.block.gate(x))*self.block.up(x))
        if c==5:return self.block.attn_norm(x) if self.reference else MathBlock.rms(x,self.block.attn_norm)
        if c==6:return torch.stack((x[:,:self.n//2],x[:,self.n//2:]),2).flatten(1,2)
        return self.head(x[:,1::2])
