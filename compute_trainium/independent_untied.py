"""Separate-projection FP32 reference for the copied, independently trained core."""
import copy
import torch
from torch import nn
from .diagnostics import traced_unroll
from .independent import IndependentModel


class IndependentUntiedModel(IndependentModel):
    def __init__(self,config,prelude=2,core=16,coda=2):
        super().__init__(config,repetitions=2,prelude=prelude,core=core,coda=coda)
        self.extra=nn.ModuleList([copy.deepcopy(self.blocks[prelude+j]) for j in range(core)])

    def load_titan_state(self,state):
        base={k:v for k,v in state.items() if k.startswith('decoder.')}
        super().load_titan_state(base);used=set(base)
        with torch.no_grad():
            for j,block in enumerate(self.extra):
                prefix=f'extra_layers.{j}.'
                pairs=[(block.q.weight,'attention.qkv_linear.wq.weight'),(block.k.weight,'attention.qkv_linear.wk.weight'),
                    (block.v.weight,'attention.qkv_linear.wv.weight'),(block.o.weight,'attention.wo.weight'),
                    (block.gate.weight,'feed_forward.w1.weight'),(block.down.weight,'feed_forward.w2.weight'),
                    (block.up.weight,'feed_forward.w3.weight'),(block.attn_norm.weight,'attention_norm.weight'),(block.ffn_norm.weight,'ffn_norm.weight')]
                for destination,suffix in pairs:
                    key=prefix+suffix;source=state[key]
                    if source.shape!=destination.shape:raise ValueError('Extra-core reference shape differs')
                    destination.copy_(source);used.add(key)
        if used!=set(state):raise ValueError('Unmapped untied state keys: '+str(set(state)-used))

    def hidden(self,tokens,attention_mask=None):
        if attention_mask is not None:raise ValueError('This reference scores unmodified causal prefixes')
        return traced_unroll(self,tokens,2,self.extra)[0]
