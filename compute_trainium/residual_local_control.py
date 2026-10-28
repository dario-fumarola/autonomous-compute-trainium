"""Parameter-matched local nonlinear readout control for causal memory branches."""
import torch
from torch import nn
from torch.nn import functional as F
from .residual_workspace import unit,ResidualWorkspace


class LocalResidualCorrection(nn.Module):
    def __init__(self,hidden_size=2048,width=128):
        super().__init__();self.input=nn.Linear(hidden_size,width,bias=False);self.up=nn.Linear(width,4*width,bias=False);self.down=nn.Linear(4*width,width,bias=False);self.output=nn.Linear(width,hidden_size,bias=False);nn.init.zeros_(self.output.weight)
    def forward(self,features,anchors,*,disabled=False):
        base=features.index_select(1,anchors)
        if disabled:return base
        state=self.input(unit(base));state=state+self.down(F.silu(self.up(unit(state))));return base+self.output(unit(state))


def make_branch(width,mode):
    if mode in ['memory_static','memory_written']:
        from .residual_written_memory import WrittenMemoryWorkspace
        return WrittenMemoryWorkspace(width=width,writes=mode=='memory_written')
    return LocalResidualCorrection(width=width) if mode=='local' else ResidualWorkspace(width=width,mode=mode)
