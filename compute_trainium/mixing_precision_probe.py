"""Precision-only alternatives for a zero-update failed-gradient diagnosis."""
import torch
from torch.nn import functional as F
from .parcae_mixing_control import MixingOnlyFutureSpanCandidate


class PrecisionMixingProbe(MixingOnlyFutureSpanCandidate):
    def __init__(self,*args,precision_mode,**kwargs):
        if precision_mode not in ['fp16','mix32','mixc32','full32']:
            raise ValueError('Explicit bounded precision variant required')
        if precision_mode=='full32':kwargs['fp16_body']=False
        super().__init__(*args,**kwargs);self.precision_mode=precision_mode

    def mix(self,state,source):
        if self.precision_mode=='fp16':return super().mix(state,source)
        with torch.autocast(state.device.type,enabled=False):
            return self.transformer.adapter(state.float(),source.float())

    def decode(self,state,tokens):
        if tokens.shape[1]!=self.future_config.context:raise ValueError('Fixed original context required')
        with torch.autocast(state.device.type,dtype=torch.float16,
                            enabled=self.fp16_body and self.precision_mode not in ['mixc32','full32']):
            state=self.transformer.C(state)
        with torch.autocast(state.device.type,dtype=torch.float16,enabled=self.fp16_body):
            for j,block in enumerate(self.transformer.coda):
                state=self._apply_block(block,state,tokens,self.config.prelude+self.config.core+j)
        with torch.autocast(state.device.type,enabled=False):
            hidden=self.transformer.ln_f(state.float())
            return F.linear(hidden,self.lm_head.weight.float()),self.future_lm_head(hidden.index_select(1,self.future_positions))
