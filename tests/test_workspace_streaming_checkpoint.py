import torch
import pytest
from compute_trainium.workspace_streaming_checkpoint import save_streaming,restore_streaming


def test_complete_streaming_state_restore_exact_and_no_overwrite(tmp_path):
    values={'model':torch.randn(4,8),'scorer':torch.randn(4,8),'adam_m':torch.randn(4,8),'adam_v':torch.rand(4,8),'step':torch.tensor(1.)}
    original={k:v.clone() for k,v in values.items()};rng=torch.get_rng_state()
    path=tmp_path/'rank0.zip';meta={'update':1,'rng':rng,'param_groups':[{'lr':1e-5,'betas':(.9,.95)}]}
    result=save_streaming(path,values.items(),meta);assert result['tensors']==5
    for v in values.values():v.add_(1.)
    restored=restore_streaming(path,values.items())
    assert all(torch.equal(v,original[n]) for n,v in values.items()) and torch.equal(restored['rng'],rng)
    assert restored['param_groups']==meta['param_groups']
    with pytest.raises(FileExistsError):save_streaming(path,values.items(),meta)
    with pytest.raises(ValueError):restore_streaming(path,list(values.items())[:-1])
