import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_static_positions import StaticPositionInterleavedWorkspace


@pytest.mark.parametrize('mode,k,n',[('ordinary',0,7),('pause',0,7),('latent',0,3),('latent',2,7),('latent',4,5)])
def test_fixed_cache_exact_prediction_and_full_gradient(mode,k,n):
    torch.manual_seed(9837)
    config=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=29,max_seq=8)
    old=InterleavedWorkspace(config,explicit_attention=True).double()
    new=StaticPositionInterleavedWorkspace(config,explicit_attention=True).double()
    new.load_state_dict(old.state_dict())
    assert set(new.state_dict())==set(old.state_dict())
    x=torch.randint(0,29,(2,n));y=torch.randint(0,29,(2,n))
    outputs=[]
    for model in (old,new):
        out=model(x,k,mode);outputs.append(out)
        F.cross_entropy(out.flatten(0,1),y.flatten()).backward()
    assert torch.equal(*outputs)
    for p,q in zip(old.parameters(),new.parameters()):assert torch.equal(p.grad,q.grad)


def test_original_sequential_reference_remains_identical():
    torch.manual_seed(9838)
    config=ModelConfig(dim=8,layers=1,heads=2,kv_heads=1,ffn_dim=16,vocab_size=17,max_seq=5)
    old=InterleavedWorkspace(config).double();new=StaticPositionInterleavedWorkspace(config).double()
    new.load_state_dict(old.state_dict());x=torch.randint(0,17,(1,5))
    assert torch.equal(old.sequential(x),new.sequential(x))
    torch.testing.assert_close(new(x,4),new.sequential(x),atol=1e-12,rtol=1e-12)
