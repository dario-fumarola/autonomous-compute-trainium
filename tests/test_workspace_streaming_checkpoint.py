import torch
import pytest
from compute_trainium.workspace_streaming_checkpoint import save_streaming,restore_streaming


@pytest.mark.parametrize('compress',[False,True])
def test_complete_streaming_state_restore_exact_and_no_overwrite(tmp_path,compress):
    values={'model':torch.randn(4,8),'scorer':torch.randn(4,8),'adam_m':torch.randn(4,8),'adam_v':torch.rand(4,8),'step':torch.tensor(1.)}
    original={k:v.clone() for k,v in values.items()};rng=torch.get_rng_state()
    path=tmp_path/'rank0.zip';meta={'update':1,'rng':rng,'param_groups':[{'lr':1e-5,'betas':(.9,.95)}]}
    result=save_streaming(path,values.items(),meta,compress=compress);assert result['tensors']==5
    for v in values.values():v.add_(1.)
    restored=restore_streaming(path,values.items())
    assert all(torch.equal(v,original[n]) for n,v in values.items()) and torch.equal(restored['rng'],rng)
    assert restored['param_groups']==meta['param_groups']
    with pytest.raises(FileExistsError):save_streaming(path,values.items(),meta)
    with pytest.raises(ValueError):restore_streaming(path,list(values.items())[:-1])


def test_complete_state_without_unused_scorer_restores_optimizer_rng_and_coverage(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from compute_trainium import workspace_native_state as state
    # State serialization only; actual DTensor shards/replay have a separate
    # four-process test. Here local tensors expose optional-scorer coverage.
    monkeypatch.setattr(state,'local_parameter',lambda p:p)
    model=torch.nn.Linear(4,4,bias=False);adam=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False)
    model(torch.randn(2,4)).sum().backward();adam.step();adam.zero_grad(set_to_none=True)
    optimizer=SimpleNamespace(named=list(model.named_parameters()),local=list(model.parameters()),adam=adam,update=1)
    rng=torch.Generator().manual_seed(7919);before={n:t.clone() for n,t in state.complete_state_tensors(model,None,optimizer)}
    rng_state=rng.get_state();torch_state=torch.get_rng_state()
    path=tmp_path/'no-scorer.zip';state.save_complete_state(path,model,None,optimizer,rng,extra={'arm':'ordinary'})
    with torch.no_grad():
        for p in model.parameters():p.add_(1.)
    adam.state.clear();rng.manual_seed(1);torch.manual_seed(2);optimizer.update=0
    state.restore_complete_state(path,model,None,optimizer,rng)
    assert optimizer.update==1 and torch.equal(rng.get_state(),rng_state) and torch.equal(torch.get_rng_state(),torch_state)
    for n,t in state.complete_state_tensors(model,None,optimizer):assert torch.equal(t,before[n])
    with pytest.raises(ValueError):state.restore_complete_state(path,model,model,optimizer,rng)
