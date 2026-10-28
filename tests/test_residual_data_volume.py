import torch
from compute_trainium.residual_workspace import ResidualWorkspace
from compute_trainium.residual_data_volume import source_indices,restore_branch


def test_source_schedule_changes_only_after_the_shared_prefix():
    for update in range(4):assert torch.equal(source_indices(update,2,8),source_indices(update,2,32))
    assert source_indices(4,2,8).tolist()==[0,1]
    assert source_indices(4,2,32).tolist()==[8,9]


def test_complete_restore_replays_next_gradient_update_and_sampling(tmp_path):
    torch.manual_seed(7381);features=torch.randn(8,7,16);head=torch.randn(19,16);targets=torch.randint(19,(8,7));m=ResidualWorkspace(16,8);o=torch.optim.AdamW(m.parameters(),lr=.001);g=torch.Generator().manual_seed(7382)
    def step(model,opt,rng,update):
        ix=source_indices(update,2,8);a=torch.randperm(7,generator=rng)[:3].sort().values;opt.zero_grad();loss=torch.nn.functional.cross_entropy((model(features[ix],a)@head.T).reshape(-1,19),targets[ix][:,a].flatten());loss.backward();grads=[v.grad.clone() for v in model.parameters()];opt.step();return loss.detach(),grads
    step(m,o,g,0);step(m,o,g,1);path=tmp_path/'state.pt';torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),rng=g.get_state(),updates=2),path)
    restored=ResidualWorkspace(16,8);other=torch.optim.AdamW(restored.parameters(),lr=99.);rng=torch.Generator();assert restore_branch(path,restored,other,rng)==2
    x,gx=step(m,o,g,2);y,gy=step(restored,other,rng,2);assert torch.equal(x,y)
    for left,right in zip(gx,gy,strict=True):assert torch.equal(left,right)
    for left,right in zip(m.parameters(),restored.parameters(),strict=True):assert torch.equal(left,right)
    assert torch.equal(g.get_state(),rng.get_state())
