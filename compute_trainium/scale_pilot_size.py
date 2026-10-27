"""Outcome-blind medium inventory to fit joint model/data scaling in the cap."""
import torch

from .parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig,matmul_ledger
from .parcae_optimizer_candidate import parameter_groups
from .scale_sharded_candidate import scale_inventory


def profiled_inventory(profile):
    if profile=='large':return scale_inventory()
    if profile!='medium':raise ValueError('Prespecified large or medium feasibility profile required')
    rows=[]
    for role,recurrent,cfg in [
        ('fixed_two',True,ParcaeCandidateConfig(dim=1536,heads=12,ffn_dim=6144,max_seq=1024)),
        ('ordinary8',False,ParcaeCandidateConfig(dim=1280,heads=10,ffn_dim=5504,prelude=2,core=4,coda=2,max_seq=1024)),
    ]:
        with torch.device('meta'):model=ParcaeCandidate(cfg,recurrent=recurrent,initialize=False)
        groups=parameter_groups(model.named_parameters(),gate_policy='release_muon')
        adam=sum(p.numel() for p in groups[0]['params'])
        muon=sum(p.numel() for g in groups[1:] for p in g['params'])
        variance=sum(len(g['params'])*max(g['params'][0].shape) for g in groups[1:])
        rows.append(dict(role=role,config=vars(cfg),parameters=cfg.parameter_count(recurrent),
            adamw_parameters=adam,muon_parameters=muon,muon_variance_elements=variance,
            master_gradient_moment_bytes=16*adam+12*muon+4*variance,
            neural_work_per_window=matmul_ledger(cfg,1024,1,recurrent=recurrent,total_steps=2 if recurrent else 1,
                backward_cap=2 if recurrent else 1)['training_matmul_flops']))
    return rows
