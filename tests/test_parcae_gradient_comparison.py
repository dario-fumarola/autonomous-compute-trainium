import pytest
import torch

from scripts.run_parcae_native_global import compare,launch_ready


def test_full_gradient_relative_error_uses_reference_norm(tmp_path):
    left,right=tmp_path/'cpu.pt',tmp_path/'native.pt'
    torch.save({'gradients':{'x':torch.tensor([3.,4.])},'loss':2.},left)
    torch.save({'gradients':{'x':torch.tensor([3.3,4.4])},'loss':2.001},right)
    row=compare(left,right)
    assert row['relative_l2']==pytest.approx(.1,abs=1e-7)
    assert row['native_minus_cpu_nll']==pytest.approx(.001)


def test_gradient_name_mismatch_cannot_be_silently_dropped(tmp_path):
    left,right=tmp_path/'cpu.pt',tmp_path/'native.pt'
    torch.save({'gradients':{'x':torch.ones(2)},'loss':2.},left)
    torch.save({'gradients':{'other':torch.ones(2)},'loss':2.},right)
    with pytest.raises(AssertionError):
        compare(left,right)


def test_ready_prerequisite_cannot_override_latest_start():
    complete={'status':'COMPLETE_COMMON_FP32_DEPTH_CONNECTION_COMPARISON'}
    assert launch_ready(complete,False,99.,100.)
    assert not launch_ready(complete,True,99.,100.)
    with pytest.raises(TimeoutError):
        launch_ready(complete,False,101.,100.)
