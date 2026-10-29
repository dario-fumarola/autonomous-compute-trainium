import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.grouped_interleaved_workspace import GroupedInterleavedWorkspace

@pytest.mark.parametrize('mode,k,n,explicit',[('ordinary',0,5,True),('pause',0,5,True),('latent',0,5,True),('latent',1,5,True),('latent',2,5,True),('latent',3,4,True),('latent',2,5,False)])
def test_reordered_history_preserves_predictions_and_all_gradients(mode,k,n,explicit):
    torch.manual_seed(9881);c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=29,max_seq=5)
    old=InterleavedWorkspace(c,explicit_attention=explicit).double()
    new=GroupedInterleavedWorkspace(c,explicit_attention=explicit).double();new.load_state_dict(old.state_dict())
    assert set(old.state_dict())==set(new.state_dict())
    x=torch.randint(0,29,(2,n));y=torch.randint(0,29,(2,n));outputs=[]
    for model in (old,new):
        out=model(x,k,mode);outputs.append(out);F.cross_entropy(out.flatten(0,1),y.flatten()).backward()
    torch.testing.assert_close(*outputs,atol=1e-11,rtol=1e-11)
    for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-11,rtol=1e-10)


def test_causality_sequential_limit_and_optimizer():
    torch.manual_seed(9882);c=ModelConfig(dim=8,layers=1,heads=2,kv_heads=1,ffn_dim=16,vocab_size=17,max_seq=5)
    old=InterleavedWorkspace(c,explicit_attention=True).double();new=GroupedInterleavedWorkspace(c,explicit_attention=True).double();new.load_state_dict(old.state_dict())
    x=torch.randint(0,17,(2,5));y=torch.randint(0,17,(2,5));changed=x.clone();changed[:,3:]=(changed[:,3:]+1)%17
    torch.testing.assert_close(new(x,2)[:,:3],new(changed,2)[:,:3],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(new(x,4),old.sequential(x),atol=1e-11,rtol=1e-11)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001,foreach=False) for m in (old,new)]
    for _ in range(3):
        for model,opt in zip((old,new),optimizers):
            opt.zero_grad();F.cross_entropy(model(x,2).flatten(0,1),y.flatten()).backward();opt.step()
        for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p,q,atol=1e-11,rtol=1e-10)


def test_registered_factory_keeps_independent_chronological_reference():
    from compute_trainium.interleaved_model_factory import make_model
    c=ModelConfig(dim=8,layers=1,heads=2,kv_heads=1,ffn_dim=16,vocab_size=17,max_seq=5)
    p=dict(model_config=c.__dict__,interleaved_implementation='grouped')
    assert type(make_model(p,explicit_attention=True)) is GroupedInterleavedWorkspace
    assert type(make_model(p)) is InterleavedWorkspace
