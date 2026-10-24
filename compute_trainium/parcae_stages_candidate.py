"""Static recurrent stages, preserving the checked Parcae candidate function.

Compile stages once rather than compiling a new whole-model unroll per depth.
Loop counts remain host integers; tensors keep their fixed window shapes.
This is a separately qualified candidate, not an active training migration.
"""
import torch
from torch.nn import functional as F

from .parcae_candidate import ParcaeCandidate
from .stable_recurrence_candidate import recurrent_tail


class ParcaeStagesCandidate(ParcaeCandidate):
    def _apply_block(self, block, state, tokens, index):
        freqs = self.freqs_cis[:, :tokens.shape[1]]
        ve = self.value_embeds[str(index)](tokens) if str(index) in self.value_embeds else None
        return block(state, freqs, ve)

    def encode(self, tokens):
        if tokens.shape[1] > self.config.max_seq:
            raise ValueError('Input exceeds declared rotary context')
        x = self.transformer.wte(tokens)
        layers = self.transformer.prelude if self.recurrent else self.transformer.h
        for i, block in enumerate(layers):
            x = self._apply_block(block, x, tokens, i)
        return self.transformer.ln_prelude(x) if self.recurrent else x

    def refine(self, state, source, tokens):
        if not self.recurrent:
            raise ValueError('Ordinary reference has no recurrent stage')
        state = self.transformer.adapter(state, source)
        for j, block in enumerate(self.transformer.core_block):
            state = self._apply_block(block, state, tokens, self.config.prelude+j)
        return state

    def decode(self, state, tokens):
        if self.recurrent:
            state = self.transformer.C(state)
            for j, block in enumerate(self.transformer.coda):
                state = self._apply_block(block, state, tokens, self.config.prelude+self.config.core+j)
        state = self.transformer.ln_f(state)
        return F.linear(state, self.lm_head.weight).float()

    def compile_stages(self, backend):
        if hasattr(self, '_execution_stages'):
            raise RuntimeError('Execution stages already bound; no silent replacement')
        names = ['encode', 'decode']+(['refine'] if self.recurrent else [])
        self._execution_stages = {name:torch.compile(getattr(self, name), backend=backend,
            fullgraph=True, dynamic=False) for name in names}

    def forward(self, tokens, state=None, *, total_steps=1, backward_cap=1):
        stages = getattr(self, '_execution_stages', None)
        encode = stages['encode'] if stages else self.encode
        decode = stages['decode'] if stages else self.decode
        source = encode(tokens)
        if self.recurrent:
            if state is None or state.shape != (*tokens.shape, self.config.dim):
                raise ValueError('Explicit correctly shaped initial state required')
            refine = stages['refine'] if stages else self.refine
            state = recurrent_tail(lambda x,e:refine(x,e,tokens), state, source, total_steps, backward_cap)
        else:
            if state is not None or total_steps != 1:
                raise ValueError('Ordinary graph has no state or repeated core')
            state = source
        return decode(state, tokens)
