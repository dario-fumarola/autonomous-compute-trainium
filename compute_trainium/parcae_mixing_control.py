"""Prospective strong cheap control; not imported by frozen campaigns.

Repeat only the input/state adapter, then execute the transformer core once.
All sampled mixing steps receive gradients. The one-step function and complete
parameter inventory equal the existing recurrent family. This control is not
claimed to be recurrent transformer computation or native-qualified training.
"""
import torch

from .future_span_math_sdpa import FutureSpanMathSDPACandidate
from .parcae_candidate import matmul_ledger


class MixingOnlyFutureSpanCandidate(FutureSpanMathSDPACandidate):
    def mix(self, state, source):
        with torch.autocast(state.device.type,dtype=torch.float16,enabled=self.fp16_body):
            return self.transformer.adapter(state,source)

    def core_once(self, state, tokens):
        with torch.autocast(state.device.type,dtype=torch.float16,enabled=self.fp16_body):
            for j,block in enumerate(self.transformer.core_block):
                state=self._apply_block(block,state,tokens,self.config.prelude+j)
        return state

    def compile_stages(self,backend):
        if hasattr(self,'_execution_stages'):
            raise RuntimeError('Execution stages already bound')
        self._execution_stages={name:torch.compile(getattr(self,name),backend=backend,
            fullgraph=True,dynamic=False) for name in ['encode','mix','core_once','decode']}

    def forward(self,tokens,state=None,*,total_steps=1,backward_cap=1):
        if (not self.recurrent or type(total_steps) is not int or type(backward_cap) is not int
                or total_steps < 1 or backward_cap != total_steps):
            raise ValueError('Explicit positive full-gradient mixing count required')
        if state is None or state.shape != (*tokens.shape,self.config.dim):
            raise ValueError('Original correctly shaped explicit initial state required')
        stages=getattr(self,'_execution_stages',None)
        def stage(name):
            return stages[name] if stages else getattr(self,name)
        source=stage('encode')(tokens)
        for _ in range(total_steps):
            state=stage('mix')(state,source)
        state=stage('core_once')(state,tokens)
        return stage('decode')(state,tokens)


def mixing_only_ledger(config,context,batch=1,*,total_steps=1,backward_cap=1):
    """Backbone work only; common auxiliary head/optimizer charged separately.

    Preserve every executed B projection, including repeated identical input
    projections. No algebraic hoisting or optimistic skipping is assumed.
    """
    if (type(total_steps) is not int or type(backward_cap) is not int
            or total_steps < 1 or backward_cap != total_steps):
        raise ValueError('Positive full-gradient mixing count required')
    base=matmul_ledger(config,context,batch,recurrent=True,total_steps=1,backward_cap=1)
    extra=2*batch*context*config.dim**2*(total_steps-1)
    return {**base,'forward_matmul_flops':base['forward_matmul_flops']+extra,
        'backward_matmul_flops':base['backward_matmul_flops']+2*extra,
        'training_matmul_flops':base['training_matmul_flops']+3*extra,
        'scope':'One complete core execution;all input-adapter repetitions and full gradients counted. Backbone/head matmuls only;auxiliary head,optimizer,pointwise work and hardware time remain separate.'}
