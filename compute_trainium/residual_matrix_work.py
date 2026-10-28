"""Count executed dense matrix scalar operations,including frozen-head backward."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class MatrixWork(TorchDispatchMode):
    def __init__(self):super().__init__();self.operations=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func in [torch.ops.aten.mm.default,torch.ops.aten.bmm.default]:self.operations+=2*args[0].numel()*args[1].shape[-1]
        return func(*args,**(kwargs or {}))
