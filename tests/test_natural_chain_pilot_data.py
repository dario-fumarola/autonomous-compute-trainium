"""Routing/noise/count software fixtures;never research source examples."""
import torch
from compute_trainium.natural_chain_pilot_data import exposure_state,task_row,answer_count
from compute_trainium.parcae_candidate import ParcaeCandidateConfig


def test_exposure_noise_is_paired_local_and_independent_of_global_rng():
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,prelude=2,core=2,coda=2,max_seq=1024)
    torch.manual_seed(3);before=torch.get_rng_state();a=exposure_state(config,17,137,128);b=exposure_state(config,17,137,128);c=exposure_state(config,18,137,128)
    assert torch.equal(a,b) and not torch.equal(a,c) and torch.equal(before,torch.get_rng_state())
    assert a.shape==(1,128,32) and float(a.abs().max())<3*(2/(5*32))**.5


def test_shared_problem_order_curriculum_and_whole_answer_counts():
    cfg=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,prelude=2,core=2,coda=2,max_seq=1024)
    masks=torch.zeros(6,128,dtype=torch.bool)
    for index in range(6):masks[index,:1+index%2]=True
    tasks=dict(inputs=torch.arange(6)[:,None].expand(6,128),targets=torch.zeros(6,128,dtype=torch.long),answer_mask=masks,
        termination_mask=torch.zeros_like(masks),view_indices=torch.tensor([[0,1],[2,3],[4,5]]),order=torch.arange(128)[None].expand(200,128)%2)
    case=dict(total_updates=200)
    assert answer_count(tasks,0,case)==192
    for update,expected in [(0,0),(50,2),(100,4),(199,4)]:
        values,index=task_row(tasks,update,0,case,cfg,137);assert index==expected and values[0][0,0]==expected
        paired,_=task_row(tasks,update,0,case,cfg,137);assert torch.equal(values[-1],paired[-1])
        other,_=task_row(tasks,update,1,case,cfg,137);assert not torch.equal(values[-1],other[-1])
