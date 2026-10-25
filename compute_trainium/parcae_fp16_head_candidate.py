"""Prospective FP16 vocabulary GEMM with FP32 masters/logits/loss.

The current frozen trainers retain their original FP32 head. This separate
candidate is CPU screening only until complete native qualification proves
gradient/update/recovery fidelity and measures any actual speed benefit.
"""
import torch
from torch.nn import functional as F

from .parcae_fp16_candidate import ParcaeFP16Candidate


class ParcaeFP16HeadCandidate(ParcaeFP16Candidate):
    def decode(self,state,tokens):
        with torch.autocast(state.device.type,dtype=torch.float16,enabled=self.fp16_body):
            if self.recurrent:
                state=self.transformer.C(state)
                for j,block in enumerate(self.transformer.coda):
                    state=self._apply_block(block,state,tokens,self.config.prelude+self.config.core+j)
        with torch.autocast(state.device.type,enabled=False):
            hidden=self.transformer.ln_f(state.float())
            # Casts preserve the tied FP32 parameter and its FP32 master
            # gradient. Manual loss scaling must cover this path too.
            return F.linear(hidden.to(torch.float16),self.lm_head.weight.to(torch.float16)).float()
