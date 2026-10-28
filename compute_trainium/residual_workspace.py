"""Small trainable workspace in frozen final-representation space.

The language model and vocabulary head are external,frozen inputs. This is a
mechanism screen,not a from-scratch or training-compute efficiency claim.
"""
import torch
from torch import nn
from torch.nn import functional as F


def unit(x):return F.rms_norm(x,(x.shape[-1],))


class RefinementCell(nn.Module):
    def __init__(self,width):
        super().__init__();self.q=nn.Linear(width,width,bias=False);self.k=nn.Linear(width,width,bias=False);self.v=nn.Linear(width,width,bias=False);self.o=nn.Linear(width,width,bias=False);self.up=nn.Linear(width,2*width,bias=False);self.down=nn.Linear(2*width,width,bias=False)
    def forward(self,state,memory,anchors):
        q=self.q(unit(state));k=self.k(unit(memory));v=self.v(unit(memory));mask=torch.arange(memory.shape[1],device=memory.device)[None,:]<=anchors[:,None]
        scores=(q@k.transpose(-1,-2))*(state.shape[-1]**-.5);weights=scores.masked_fill(~mask,float('-inf')).softmax(-1)
        state=state+self.o(weights@v);return state+self.down(F.silu(self.up(unit(state))))


class ResidualWorkspace(nn.Module):
    def __init__(self,hidden_size=2048,width=128,*,mode='two'):
        super().__init__()
        if mode not in ['linear','one','two']:raise ValueError('Registered mechanism/control required')
        self.mode=mode;self.input=nn.Linear(hidden_size,width,bias=False);self.cell=None if mode=='linear' else RefinementCell(width);self.output=nn.Linear(width,hidden_size,bias=False);nn.init.zeros_(self.output.weight)
    def forward(self,features,anchors,*,disabled=False):
        base=features.index_select(1,anchors)
        if disabled:return base
        memory=self.input(unit(features));state=memory.index_select(1,anchors)
        if self.cell is not None:
            for _ in range(1 if self.mode=='one' else 2):state=self.cell(state,memory,anchors)
        delta=self.output(state if self.cell is None else unit(state))
        return base+delta
