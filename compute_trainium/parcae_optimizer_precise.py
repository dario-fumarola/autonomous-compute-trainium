"""Explicit FP32-polar optimizer variant; this changes the released recipe.

Derived from the MIT Parcae Muon implementation, copyright(c)2026 Hayden
Prairie; see third_party/parcae_reference/LICENSE. Preserves masters, grouping,
five coefficients, momentum, variance normalization and cautious decay. Every
polar intermediate is FP32. Requires its own CPU/native/recovery/profile gates.
"""
import ast
import inspect
import textwrap

import torch

from .parcae_optimizer_candidate import POLAR_COEFFS, ParcaeOptimizerCandidate


def muon_tensor_step_fp32(gradient, parameters, momentum_state, variance_state,
                          momentum, lr, decay, beta2, steps, reduction_dimension):
    momentum_value = momentum.to(gradient.dtype)
    momentum_state.lerp_(gradient, 1-momentum_value)
    update = gradient.lerp_(momentum_state, momentum_value)
    polar = update.float()
    polar = polar/(polar.norm(dim=(-2,-1),keepdim=True)*1.02+1e-6)
    for a,b,c in POLAR_COEFFS[:steps]:
        if gradient.size(-2)>gradient.size(-1):
            gram=polar.mT@polar
            polar=a*polar+polar@(b*gram+c*(gram@gram))
        else:
            gram=polar@polar.mT
            polar=a*polar+(b*gram+c*(gram@gram))@polar
    beta=beta2.to(polar.dtype)
    mean=polar.float().square().mean(dim=reduction_dimension,keepdim=True)
    norm=(mean.sum(dim=(-2,-1),keepdim=True)*polar.size(reduction_dimension)).sqrt()
    variance_state.lerp_(mean.to(variance_state.dtype),(1-beta).to(variance_state.dtype))
    scale=variance_state.clamp_min(1e-10).rsqrt()
    new_norm=((mean*polar.size(reduction_dimension))*scale.float().square()).sum(dim=(-2,-1),keepdim=True).sqrt()
    update=polar*(scale*(norm/new_norm.clamp_min(1e-10))).to(polar.dtype)
    rate,wd=lr.to(update.dtype),decay.to(update.dtype)
    cautious=(update*parameters)>=0
    parameters.sub_(rate*update+rate*wd*parameters*cautious)


class ParcaePreciseOptimizer(ParcaeOptimizerCandidate):
    def __init__(self,model,*,gate_policy,lr=.008,muon_lr=.008,fused_adamw=False,muon_backend=None):
        super().__init__(model,gate_policy=gate_policy,lr=lr,muon_lr=muon_lr,
                         fused_adamw=fused_adamw,muon_backend=None)
        self.muon.tensor_step=(torch.compile(muon_tensor_step_fp32,backend=muon_backend,fullgraph=True,dynamic=False)
                               if muon_backend else muon_tensor_step_fp32)
        self.recipe.update(identity='parcae_muon_adamw_fp32_polar_v2',specified_muon_polar_dtype='fp32')


def independent_fp32_reference(module):
    """One audited dtype substitution in pinned upstream code, not our kernel.

    Decorator removal only executes the independent CPU algorithm eagerly.
    This produces a separate function and never edits the vendored source.
    """
    original=getattr(module.muon_step,'_torchdynamo_orig_callable',module.muon_step)
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1:raise ValueError('Pinned reference function inventory changed')
    functions[0].decorator_list=[]
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='bfloat16']
    if len(calls)!=1 or calls[0].args or calls[0].keywords or not isinstance(calls[0].func.value,ast.Name) or calls[0].func.value.id!='g':
        raise ValueError('Expected exactly the pinned upstream g.bfloat16() conversion')
    calls[0].func.attr='float'
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<independent-upstream-fp32-polar>','exec'),namespace)
    return namespace[functions[0].name]
