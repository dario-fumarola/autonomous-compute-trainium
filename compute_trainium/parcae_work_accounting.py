"""Explicit executed matmul work, including replicated Muon updates."""
from .parcae_candidate import matmul_ledger
from .parcae_optimizer_candidate import parameter_groups


def muon_update_matmuls(named_parameters, *, gate_policy, steps=5):
    """One replica's polar-iteration matmuls; AdamW/pointwise work is separate."""
    if type(steps) is not int or not 1 <= steps <= 5:
        raise ValueError('One to five polar iterations required')
    groups=parameter_groups(named_parameters,gate_policy=gate_policy,steps=steps)
    total=0
    for group in groups:
        if group['kind']!='muon':continue
        for parameter in group['params']:
            rows,columns=parameter.shape
            small,large=min(rows,columns),max(rows,columns)
            # Gram, Gram squared, and matrix-times-polynomial, per iteration.
            total+=steps*(4*large*small*small+2*small**3)
    return total


def global_update_work(model,context,world,batch,accumulation,*,gate_policy,
                       forward_steps=1,backward_cap=1):
    if any(type(v) is not int or v<1 for v in (context,world,batch,accumulation)):
        raise ValueError('Positive integer execution geometry required')
    neural=matmul_ledger(model.config,context,world*batch*accumulation,
        recurrent=model.recurrent,total_steps=forward_steps,backward_cap=backward_cap)
    optimizer=muon_update_matmuls(model.named_parameters(),gate_policy=gate_policy)
    duplicated=optimizer*world
    return {'model_forward_backward_matmuls':neural['training_matmul_flops'],
        'one_replica_optimizer_matmuls':optimizer,
        'executed_optimizer_matmuls_all_replicas':duplicated,
        'total_executed_matmuls':neural['training_matmul_flops']+duplicated,
        'optimizer_over_model_matmul_ratio':duplicated/neural['training_matmul_flops'],
        'world':world,'batch':batch,'accumulation':accumulation,
        'global_windows':world*batch*accumulation,'positions_per_window':context,
        'scope':'Executed dense matrix multiplies, FMA=2, full-square attention. Counts redundant optimizer work on every DDP replica. Excludes embedding gathers, elementwise/AdamW work, communication, idle time and compilation; not chip time or a hardware counter.'}
