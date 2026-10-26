import numpy as np
import pytest
from compute_trainium.lean_baseline_science import update_work,validate_case,EXECUTION,evaluation_window_key
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import mixing_only_ledger


def test_cheap_work_counts_two_mixes_and_only_one_core():
    config=ParcaeCandidateConfig(vocab_size=32000);depth=np.full((3,16,32,2),2,dtype=np.uint16)
    result=update_work('mixing',config,depth,recurrent=True,context=1024,world=32,optimizer_per_replica=123)
    expected=512*mixing_only_ledger(config,1024,total_steps=2,backward_cap=2)['training_matmul_flops']+32*123
    assert np.all(result==expected)
    depth[0,0,0,1]=1
    with pytest.raises(ValueError):update_work('mixing',config,depth,recurrent=True,context=1024,world=32,optimizer_per_replica=123)


def test_science_variant_cannot_change_qualified_geometry_or_inventory():
    plan=dict(model_variant='ordinary8',precision='fp16',accumulation=16,execution_variant=EXECUTION)
    case=dict(arm='ordinary',parameters=141733120)
    assert validate_case(plan,case)=='ordinary8'
    with pytest.raises(ValueError):validate_case(plan,{**case,'parameters':141963840})


def test_readout_noise_uses_window_position_and_fixed_full_draw_before_slice():
    import torch
    from compute_trainium.future_span_evaluation import window_state
    from compute_trainium.parcae_training_candidate import initial_state
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64)
    for document in ['document-a','unrelated-document']:
        key=evaluation_window_key(document,2)
        assert key==2
        assert torch.equal(initial_state(config,key,1009)[:,:17],window_state(config,2,1009,17))
    with pytest.raises(ValueError):evaluation_window_key('document',-1)
