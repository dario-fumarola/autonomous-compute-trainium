"""Integer nominal matrix ledgers; long fits exceed signed 64-bit totals."""
import torch

from .parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig,matmul_ledger
from .parcae_work_accounting import muon_update_matmuls


def fixed_update_work(config,role,*,global_windows=512):
    if role not in ('fixed_two','ordinary8') or type(global_windows) is not int or global_windows<1:
        raise ValueError('Fixed qualified role and positive global window count required')
    if isinstance(config,dict):config=ParcaeCandidateConfig(**config)
    recurrent=role=='fixed_two';depth=2 if recurrent else 1
    with torch.device('meta'):model=ParcaeCandidate(config,recurrent=recurrent,initialize=False)
    neural=int(matmul_ledger(config,1024,global_windows,recurrent=recurrent,
        total_steps=depth,backward_cap=depth)['training_matmul_flops'])
    owner=int(muon_update_matmuls(model.named_parameters(),gate_policy='release_muon'))
    return dict(neural_per_update=neural,owner_optimizer_per_update=owner,total_per_update=neural+owner,
        global_windows=global_windows,scope='Nominal dense neural/attention/readout and once-per-group owner Muon matrices. Elementwise/AdamW,collectives,padding and compiler recomputation require separate cost/time reporting.')


def total_work(per_update,updates):
    if type(per_update) is not int or type(updates) is not int or min(per_update,updates)<1:
        raise ValueError('Positive Python integers required; do not silently overflow NumPy counters')
    return per_update*updates


def work_matched_updates(candidate_per_update,candidate_updates,control_per_update):
    target=total_work(candidate_per_update,candidate_updates)
    if type(control_per_update) is not int or control_per_update<1:raise ValueError('Positive control work required')
    updates=target//control_per_update
    if updates<1:raise ValueError('Common budget cannot fit a complete control update')
    return dict(updates=updates,candidate_total=target,control_total=updates*control_per_update,
        unmatched_work=target-updates*control_per_update)
