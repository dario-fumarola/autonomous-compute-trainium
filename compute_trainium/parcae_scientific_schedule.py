"""Integer work/source endpoints and branch-compatible schedules, no launcher."""
import math

import numpy as np

from .parcae_candidate import matmul_ledger


REFERENCE_BUDGET=1_858_147_163_150_745_600


def update_work(config,depths,*,recurrent,context,world,optimizer_per_replica):
    """Actual per-sequence forward/backward tables plus every DDP optimizer.

    Dense semantic matmuls include full-square attention and tied output head.
    Pointwise/embedding/communication and hardware tile padding are separate;
    this is not a hardware FLOP counter or a chip-time equivalence claim.
    """
    values=np.asarray(depths)
    if (values.ndim!=4 or values.shape[-1]!=2 or values.shape[-2]!=world
        or not np.issubdtype(values.dtype,np.integer) or min(values.shape)<1
        or type(world) is not int or world<1 or type(optimizer_per_replica) is not int
        or optimizer_per_replica<0):
        raise ValueError('Integer update/micro/rank/(forward,backward) table required')
    if np.any(values<1) or np.any(values[...,1]>values[...,0]):
        raise ValueError('Every differentiated tail must be positive and no longer than its forward count')
    windows=values.shape[1]*world
    base=matmul_ledger(config,context,1,recurrent=recurrent,total_steps=1,backward_cap=1)['training_matmul_flops']
    if recurrent:
        repeated=matmul_ledger(config,context,1,total_steps=2,backward_cap=1)['training_matmul_flops']-base
        once=base-3*repeated
        counts=(values[...,0].astype(np.int64)+2*values[...,1].astype(np.int64)).sum(axis=(1,2))
        if int(counts.max())*repeated+windows*once+optimizer_per_replica*world>np.iinfo(np.int64).max:
            raise OverflowError('Per-update work exceeds signed64 ledger')
        neural=counts*repeated+windows*once
    else:
        if np.any(values!=1):raise ValueError('Ordinary executes exactly one fully differentiated pass')
        if windows*base+optimizer_per_replica*world>np.iinfo(np.int64).max:
            raise OverflowError('Per-update work exceeds signed64 ledger')
        neural=np.full(values.shape[0],windows*base,dtype=np.int64)
    result=neural+optimizer_per_replica*world
    if np.any(result<=0):raise OverflowError('Invalid or overflowed per-update work')
    return result


def matched_endpoint(work,budget=REFERENCE_BUDGET):
    values=np.asarray(work)
    if values.ndim!=1 or not len(values) or not np.issubdtype(values.dtype,np.integer) or np.any(values<=0):
        raise ValueError('Positive integer update-work vector required')
    if type(budget) is not int or budget<=0:raise ValueError('Positive integer budget required')
    # Python sums detect signed64 overflow before cumulative NumPy arithmetic.
    total=sum(map(int,values))
    if total>np.iinfo(np.int64).max:raise OverflowError('Cumulative work exceeds signed64 ledger')
    cumulative=np.cumsum(values,dtype=np.int64)
    if total<budget:raise ValueError('Provision a next update to establish a matched-work endpoint')
    updates=int(np.searchsorted(cumulative,budget,side='right'))
    if not updates:raise ValueError('Budget cannot fund one complete update')
    actual=int(cumulative[updates-1])
    return {'updates':updates,'executed_matmuls':actual,'budget_matmuls':budget,
            'unused_budget_matmuls':budget-actual,'relative_shortfall':(budget-actual)/budget,
            'next_update_would_exceed_budget':updates==len(values) or int(cumulative[updates])>budget}


def source_endpoint(prefix_counts,updates,global_windows):
    values=np.asarray(prefix_counts)
    cursor=updates*global_windows
    if (values.ndim!=2 or values.shape[1]!=3 or not np.issubdtype(values.dtype,np.integer)
        or type(updates) is not int or updates<1 or type(global_windows) is not int
        or global_windows<1 or cursor>=len(values)):
        raise ValueError('Provisioned three-domain prefix counts and complete updates required')
    counts=values[cursor].astype(np.int64)-values[0].astype(np.int64)
    if np.any(counts<0):raise ValueError('Negative source exposure')
    return {'source_cursor':cursor,'domain_targets':list(map(int,counts)),'real_targets':int(counts.sum())}


def cooldown_start(endpoint_updates,fraction=.5):
    if type(endpoint_updates) is not int or endpoint_updates<2 or not 0<fraction<1:
        raise ValueError('Nonempty stable and cooldown phases required')
    return max(1,min(endpoint_updates-1,math.floor(endpoint_updates*(1-fraction))))


def schedule(exposure,*,warmup_targets,cooldown_start_targets,end_targets,peak_lr):
    """Absolute warmup supports shared stable trunks and independent cooldowns.

    A caller must register the absolute warmup and exact endpoint table before
    any fit. This helper chooses no exposure, rate, model or scientific outcome.
    """
    if not (0<exposure<=end_targets and 0<warmup_targets<cooldown_start_targets<end_targets
            and math.isfinite(peak_lr) and peak_lr>0):
        raise ValueError('Ordered positive source milestones and finite peak rate required')
    warm=min(1.,exposure/warmup_targets)
    progress=max(0.,(exposure-cooldown_start_targets)/(end_targets-cooldown_start_targets))
    return {'lr':peak_lr*warm*.5*(1+math.cos(math.pi*progress)),
            'muon_momentum':.85+.10*warm,'muon_cautious_decay':.2*(1-progress)}
