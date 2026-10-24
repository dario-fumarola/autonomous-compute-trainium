"""Separate FP16-body candidate; the qualified FP32 stages stay unchanged."""
import torch
from torch.nn import functional as F

from .parcae_stages_candidate import ParcaeStagesCandidate


class ParcaeFP16Candidate(ParcaeStagesCandidate):
    def __init__(self, *args, fp16_body=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_body = fp16_body

    def encode(self, tokens):
        with torch.autocast(tokens.device.type, dtype=torch.float16, enabled=self.fp16_body):
            return super().encode(tokens)

    def refine(self, state, source, tokens):
        with torch.autocast(state.device.type, dtype=torch.float16, enabled=self.fp16_body):
            return super().refine(state, source, tokens)

    def decode(self, state, tokens):
        with torch.autocast(state.device.type, dtype=torch.float16, enabled=self.fp16_body):
            if self.recurrent:
                state = self.transformer.C(state)
                for j, block in enumerate(self.transformer.coda):
                    state = self._apply_block(block, state, tokens, self.config.prelude+self.config.core+j)
        with torch.autocast(state.device.type, enabled=False):
            state = self.transformer.ln_f(state.float())
            return F.linear(state, self.lm_head.weight.float())
