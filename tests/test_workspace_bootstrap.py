import copy
import pytest
import torch
from scripts.bootstrap_workspace_reader import exact_equal,learning_rate


def test_restore_check_detects_nested_moment_value_dtype_and_missing_metadata():
    state={'model':{'weight':torch.tensor([1.,2.])},'optimizer':{'state':{0:{'exp_avg':torch.tensor([.1,.2])}}},
           'rng':torch.tensor([1,2],dtype=torch.uint8),'updates':30}
    assert exact_equal(state,copy.deepcopy(state))
    changed=copy.deepcopy(state);changed['optimizer']['state'][0]['exp_avg'][1]=.3
    assert not exact_equal(state,changed)
    changed=copy.deepcopy(state);changed['rng']=changed['rng'].long()
    assert not exact_equal(state,changed)
    changed=copy.deepcopy(state);del changed['updates']
    assert not exact_equal(state,changed)


def test_bounded_bootstrap_reaches_zero_and_cannot_extend():
    assert learning_rate(1)==pytest.approx(1e-5/3)
    assert learning_rate(3)==learning_rate(27)==1e-5
    assert learning_rate(27)>learning_rate(28)>learning_rate(29)>learning_rate(30)==0
    for step in [0,31]:
        with pytest.raises(ValueError):learning_rate(step)
