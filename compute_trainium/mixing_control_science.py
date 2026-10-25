"""Prospective identity/work/reference adapters for the cheap control only."""
from types import MethodType
import numpy as np

from .future_span_scientific_state import signature as parent_signature,worker_sources as parent_sources
from .parcae_mixing_control import mixing_only_ledger

VARIANT='all_mixes_then_one_core_fullgrad'


def signature(plan_sha,plan,case):
    if plan.get('model_variant')!=VARIANT or case['arm']!='recurrent' or case['objective_weight']!=0.:
        raise ValueError('Explicit sampled-mixing NTP control required')
    return {**parent_signature(plan_sha,plan,case),'model_variant':VARIANT,
            'transformer_core_executions_per_window':1}


def worker_sources(plan):
    if plan.get('model_variant')!=VARIANT:
        raise ValueError('Explicit cheap-control scientific variant required')
    return parent_sources(plan)


def matrix_work(config,span,depths,*,world,optimizer_per_replica):
    values=np.asarray(depths)
    if (values.ndim!=4 or values.shape[-1]!=2 or values.shape[2]!=world
            or not np.issubdtype(values.dtype,np.integer) or (values<1).any()
            or not np.array_equal(values[...,0],values[...,1])):
        raise ValueError('Exact positive full-gradient mixing schedule required')
    one=mixing_only_ledger(config,span.context)['training_matmul_flops']
    auxiliary=6*len(span.positions)*config.dim*config.vocab_size
    extra_projection=6*span.context*config.dim**2
    windows=values.shape[1]*values.shape[2]
    fixed=windows*(one+auxiliary)+world*optimizer_per_replica
    counts=(values[...,0].astype(np.int64)-1).sum(axis=(1,2))
    maximum=fixed+int(counts.max())*extra_projection
    if fixed<0 or maximum>np.iinfo(np.int64).max:
        raise OverflowError('Control update work exceeds signed64')
    return fixed+counts*extra_projection


def make_reference(config,recurrent):
    """Independent pinned upstream prediction-only modules,without an aux head."""
    if not recurrent:raise ValueError('This control retains the full recurrent-family inventory')
    from scripts.audit_parcae_precise_training_point import make_reference as original
    model=original(config,True)
    def iterate(self,source,freqs,mask,num_steps_pair=None):
        if num_steps_pair is None or len(num_steps_pair)!=2:
            raise ValueError('Explicit supported mixing depth required')
        prefix,steps=map(int,num_steps_pair)
        if prefix!=0 or steps<1:raise ValueError('All mixing steps must receive gradients')
        state=self.initialize_state(source)
        for _ in range(steps):state=self.transformer.adapter(state,source)
        for j,block in enumerate(self.transformer.core_block):
            key=str(self.config.n_layers_in_prelude+j)
            value=self.value_embeds[key](self._current_input_ids) if key in self.value_embeds else None
            state=block(state,freqs,mask,ve=value)
        return state,prefix,steps,state.detach()
    model.iterate_forward=MethodType(iterate,model)
    return model
