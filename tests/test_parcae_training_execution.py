import pytest
import torch

from scripts.run_parcae_training_qualification import finite_state,launch_policy


def test_training_queue_waits_for_prior_precision_gate_and_never_launches_late():
    g={'status':'PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS'}
    o={'status':'PASS_PAIRED_NATIVE_PARCAE_OPTIMIZER_AND_RECOVERY'}
    f={'status':'PASS_PAIRED_NATIVE_PARCAE_FP16_BODY_GRADIENTS'}
    assert launch_policy(g,o,None,False,1,2) is None
    assert launch_policy(g,o,f,True,1,2) is None
    assert launch_policy(g,o,f,False,1,2)==['fp32','fp16']
    assert launch_policy(g,o,{'status':'STOPPED_NATIVE_PARCAE_FP16_BODY_QUALIFICATION'},False,1,2)==['fp32']
    with pytest.raises(RuntimeError):launch_policy(g,{'status':'STOPPED'},f,False,1,2)
    with pytest.raises(TimeoutError):launch_policy(g,o,f,False,3,2)


def test_complete_optimizer_finiteness_checks_nested_state_not_just_model():
    finite_state({'model':torch.ones(2),'optimizer':{'muon':{'state':[torch.ones(2),1]}}})
    with pytest.raises(ValueError,match='Nonfinite'):
        finite_state({'model':torch.ones(2),'optimizer':{'muon':{'state':[torch.tensor(float('nan'))]}}})
