import torch
from compute_trainium.residual_workspace import ResidualWorkspace
from compute_trainium.residual_local_control import LocalResidualCorrection
from compute_trainium.residual_data_volume import restore_branch


def test_local_capacity_identity_and_no_extra_cross_position_access():
    torch.manual_seed(7285);local=LocalResidualCorrection(16,8)
    torch.manual_seed(7285);memory=ResidualWorkspace(16,8,mode='one')
    assert sum(p.numel() for p in local.parameters())==sum(p.numel() for p in memory.parameters())
    assert torch.equal(local.input.weight,memory.input.weight)
    x=torch.randn(2,9,16);anchors=torch.tensor([2,6]);assert torch.equal(local(x,anchors),x[:,anchors])
    with torch.no_grad():local.output.weight.normal_(std=.03)
    changed=x.clone();changed[:,[0,1,3,4,5,7,8]]+=100
    assert torch.equal(local(x,anchors),local(changed,anchors))
    local(x,anchors).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in local.parameters())


def test_local_complete_resume_replays_update(tmp_path):
    torch.manual_seed(7383);x=torch.randn(2,9,16);m=LocalResidualCorrection(16,8);opt=torch.optim.AdamW(m.parameters(),lr=.001);g=torch.Generator().manual_seed(7384)
    def step(model,optimizer,rng):
        optimizer.zero_grad();a=torch.randperm(9,generator=rng)[:3].sort().values;loss=model(x,a).square().mean();loss.backward();grads=[p.grad.clone() for p in model.parameters()];optimizer.step();return loss.detach(),grads
    step(m,opt,g);path=tmp_path/'state.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=g.get_state(),updates=1),path)
    other=LocalResidualCorrection(16,8);op=torch.optim.AdamW(other.parameters(),lr=99.);rng=torch.Generator();assert restore_branch(path,other,op,rng)==1
    a,ga=step(m,opt,g);b,gb=step(other,op,rng);assert torch.equal(a,b)
    for left,right in zip(ga,gb,strict=True):assert torch.equal(left,right)
    for left,right in zip(m.parameters(),other.parameters(),strict=True):assert torch.equal(left,right)
    assert torch.equal(g.get_state(),rng.get_state())
