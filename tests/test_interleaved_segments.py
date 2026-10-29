import pytest
import torch
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_objective import target_loss_sum
from compute_trainium.interleaved_segments import SegmentedObjective
from compute_trainium.model import ModelConfig


@pytest.mark.parametrize('compile_parts',[False,True])
def test_pass_boundaries_preserve_full_gradient_and_adamw_trajectory(compile_parts):
    torch.manual_seed(9847)
    c=ModelConfig(dim=8,layers=1,heads=2,kv_heads=1,ffn_dim=16,vocab_size=17,max_seq=8)
    old=InterleavedWorkspace(c,explicit_attention=True).double()
    new=InterleavedWorkspace(c,explicit_attention=True).double();new.load_state_dict(old.state_dict())
    compiler=(lambda m:torch.compile(m,backend='aot_eager',fullgraph=True)) if compile_parts else (lambda m:m)
    objective=SegmentedObjective(new,2,compile_module=compiler)
    x=torch.randint(0,17,(2,5));y=torch.randint(0,17,(2,5));mask=torch.ones_like(y);mask[0,-2:]=0
    optimizers=[torch.optim.AdamW(model.parameters(),lr=.001,foreach=False) for model in (old,new)]
    for _ in range(3):
        for optimizer in optimizers:optimizer.zero_grad()
        a=target_loss_sum(old,x,y,mask,2)/mask.sum();b=objective(x,y,mask)/mask.sum()
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        a.backward();b.backward()
        for p,q in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-12,rtol=1e-10)
        for optimizer in optimizers:optimizer.step()
        for p,q in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(p,q,atol=1e-12,rtol=1e-10)
    assert set(old.state_dict())==set(new.state_dict())
