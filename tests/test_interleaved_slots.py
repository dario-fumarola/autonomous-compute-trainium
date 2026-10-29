import copy
import pytest
import torch
from torch.nn import functional as F
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_slots import SlotInterleavedWorkspace
from compute_trainium.model import ModelConfig


@pytest.mark.parametrize('mode,k',[('ordinary',0),('pause',0),('latent',0),('latent',2)])
def test_slot_axis_forward_gradient_and_update_equivalence(mode,k):
    torch.manual_seed(9831)
    config=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=29,max_seq=8)
    old=InterleavedWorkspace(config,explicit_attention=True).double()
    new=SlotInterleavedWorkspace(config,explicit_attention=True).double()
    new.load_state_dict(old.state_dict())
    x=torch.randint(0,29,(2,7));y=torch.randint(0,29,(2,7))
    mask=torch.ones_like(y);mask[0,-2:]=0
    optimizers=[torch.optim.AdamW(model.parameters(),lr=.001,foreach=False) for model in (old,new)]
    for _ in range(3):
        outputs=[]
        for model,optimizer in zip((old,new),optimizers):
            optimizer.zero_grad();out=model(x,k,mode);outputs.append(out)
            loss=(F.cross_entropy(out.flatten(0,1),y.flatten(),reduction='none').view_as(y)*mask).sum()/mask.sum()
            loss.backward()
        torch.testing.assert_close(*outputs,atol=1e-12,rtol=1e-12)
        for p,q in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-12,rtol=1e-10)
        for optimizer in optimizers:optimizer.step()
        for p,q in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(p,q,atol=1e-12,rtol=1e-10)


def test_slot_vjp_handles_each_stream_without_crossing_positions():
    x=torch.randn(2,10,3,dtype=torch.float64,requires_grad=True)
    source,latent=SlotInterleavedWorkspace.source_and_latent(x)
    torch.testing.assert_close(source,x[:,::2],atol=0,rtol=0)
    torch.testing.assert_close(latent,x[:,1::2],atol=0,rtol=0)
    (source.sum()+2*latent.sum()).backward()
    assert torch.equal(x.grad[:,::2],torch.ones_like(source))
    assert torch.equal(x.grad[:,1::2],2*torch.ones_like(latent))
