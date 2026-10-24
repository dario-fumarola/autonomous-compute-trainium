import torch
from torch.utils.flop_counter import FlopCounterMode

from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.parcae_work_accounting import global_update_work,muon_update_matmuls


def test_optimizer_ledger_matches_actual_matrix_operations():
    torch.manual_seed(217)
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    m=ParcaeCandidate(c,initialize=False)
    opt=ParcaeOptimizerCandidate(m,gate_policy='release_muon')
    for p in m.parameters():p.grad=torch.randn_like(p)*.1
    with FlopCounterMode(display=False) as counter:
        opt.step()
    assert counter.get_total_flops()==muon_update_matmuls(m.named_parameters(),gate_policy='release_muon')


def test_ddp_replication_increases_optimizer_work_at_same_global_batch():
    with torch.device('meta'):
        m=ParcaeCandidate(ParcaeCandidateConfig(),initialize=False)
    half=global_update_work(m,1024,32,1,8,gate_policy='release_muon',forward_steps=4,backward_cap=2)
    full=global_update_work(m,1024,64,1,4,gate_policy='release_muon',forward_steps=4,backward_cap=2)
    assert half['model_forward_backward_matmuls']==full['model_forward_backward_matmuls']
    assert full['executed_optimizer_matmuls_all_replicas']==2*half['executed_optimizer_matmuls_all_replicas']
