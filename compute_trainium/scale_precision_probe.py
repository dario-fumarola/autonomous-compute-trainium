"""Unchanged-weights precision interventions; no training admission."""
import torch
from torch.nn import functional as F

from .parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate


class FP32BoundaryCandidate(ParcaePrimitiveSDPACandidate):
    """Preserve full recurrent algebra; execute injection/readout in FP32."""
    def refine(self,state,source,tokens):
        with torch.autocast(state.device.type,enabled=False):
            state=self.transformer.adapter(state.float(),source.float())
        with torch.autocast(state.device.type,dtype=torch.float16,enabled=self.fp16_body):
            for j,block in enumerate(self.transformer.core_block):
                state=self._apply_block(block,state,tokens,self.config.prelude+j)
        return state

    def decode(self,state,tokens):
        with torch.autocast(state.device.type,enabled=False):state=self.transformer.C(state.float())
        with torch.autocast(state.device.type,dtype=torch.float16,enabled=self.fp16_body):
            for j,block in enumerate(self.transformer.coda):
                state=self._apply_block(block,state,tokens,self.config.prelude+self.config.core+j)
        with torch.autocast(state.device.type,enabled=False):
            hidden=self.transformer.ln_f(state.float());return F.linear(hidden,self.lm_head.weight.float())


def probe_policy(variant):
    if variant=='scaled16':return ParcaePrimitiveSDPACandidate,True,65536
    if variant=='boundary32':return FP32BoundaryCandidate,True,8192
    if variant=='full32':return ParcaePrimitiveSDPACandidate,False,8192
    raise ValueError('Only the three predeclared numerical interventions are supported')
