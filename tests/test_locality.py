import torch

from scripts.diagnose_pause_locality import bins


def test_locality_bins_cover_each_target_once_and_reset_at_block():
    index=bins(1024)
    assert torch.bincount(index,minlength=6).tolist()==[127,8,49,168,224,448]
    assert index[126].item()==0
    assert index[127].item()==1
    assert index[128].item()==2
    assert index[255].item()==1
    assert index[1023].item()==1
