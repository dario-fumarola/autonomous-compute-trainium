"""Random numerical tokens are fixtures, never research observations."""
import pytest
import torch

from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
from compute_trainium.chunked_path_intervention import readout
from compute_trainium.model import ModelConfig


def fixture():
    torch.manual_seed(122)
    c=ModelConfig(vocab_size=32,dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,max_seq=12)
    m=ChunkedPersistentWorkspace(c).double().eval()
    return m,torch.randint(0,32,(2,9))


@pytest.mark.parametrize('mode',['latent','pause'])
def test_authentic_path_and_two_switches_have_exact_identities(mode):
    model,x=fixture()
    with torch.inference_mode():
        actual=model.cached(x,3,1,mode)
        authentic=readout(model,x,3,mode=mode)
        boundary=readout(model,x,3,mode=mode,source_boundary=True)
        blocked=readout(model,x,3,mode=mode,block_memory=True)
        both=readout(model,x,3,mode=mode,source_boundary=True,block_memory=True)
        ordinary=model(x,steps=0)
    torch.testing.assert_close(authentic,actual,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(both,ordinary,atol=1e-12,rtol=1e-12)
    keep=torch.tensor([0,1,3,4,6,7])
    torch.testing.assert_close(boundary[:,keep],authentic[:,keep],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(blocked[:,:3],authentic[:,:3],atol=1e-12,rtol=1e-12)
    assert not torch.allclose(boundary[:,2::3],authentic[:,2::3],atol=1e-10,rtol=0)


@pytest.mark.parametrize('source_boundary,block_memory',[(False,False),(True,False),(False,True),(True,True)])
def test_future_suffix_never_changes_earlier_intervention_predictions(source_boundary,block_memory):
    model,x=fixture();changed=x.clone();changed[:,6:]=(changed[:,6:]+1)%32
    with torch.inference_mode():
        a=readout(model,x,3,source_boundary=source_boundary,block_memory=block_memory)
        b=readout(model,changed,3,source_boundary=source_boundary,block_memory=block_memory)
        short=readout(model,x[:,:2],3,source_boundary=source_boundary,block_memory=block_memory)
        ordinary=model(x[:,:2],steps=0)
    torch.testing.assert_close(a[:,:6],b[:,:6],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(short,ordinary,atol=1e-12,rtol=1e-12)
