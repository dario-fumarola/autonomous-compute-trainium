"""Numerical software fixtures only; no research examples or model scores."""
import pytest
import torch

from compute_trainium.chunked_memory_intervention import memory_mask, readout
from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
from compute_trainium.model import ModelConfig


def model(explicit=False):
    torch.manual_seed(143)
    return ChunkedPersistentWorkspace(ModelConfig(dim=16,layers=2,heads=4,
        kv_heads=2,ffn_dim=24,vocab_size=31,max_seq=8),explicit_attention=explicit).double()


def test_mask_removes_only_earlier_latent_keys_and_retains_self():
    kinds=[False,False,True,False,False,True]
    a=memory_mask(kinds);b=memory_mask(kinds,block_earlier_latents=True)
    expected=a.clone();expected[3:,2]=False
    assert torch.equal(b,expected)
    assert bool(b.diag().all()) and not bool(b.triu(1).any())


@pytest.mark.parametrize('explicit',[False,True])
def test_authentic_readout_is_existing_model_and_intervention_starts_after_slot(explicit):
    m=model(explicit);x=torch.tensor([[2,5,8,11,14]])
    a=readout(m,x,2);b=m.full_reference(x,2,1)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
    changed=readout(m,x,2,block_earlier_latents=True)
    torch.testing.assert_close(changed[:,:2],a[:,:2],atol=0,rtol=0)
    assert float((changed[:,2:]-a[:,2:]).detach().abs().max())>1e-6
    for length in range(1,6):
        short=readout(m,x[:,:length],2,block_earlier_latents=True)
        torch.testing.assert_close(short,changed[:,:length],atol=1e-12,rtol=1e-10)


def test_no_latent_slots_gives_identical_intervention_and_no_future_leakage():
    m=model();x=torch.tensor([[2,5,8,11,14]])
    assert torch.equal(readout(m,x,8),readout(m,x,8,block_earlier_latents=True))
    altered=x.clone();altered[:,3:]=30
    a=readout(m,x,2,block_earlier_latents=True)
    b=readout(m,altered,2,block_earlier_latents=True)
    torch.testing.assert_close(a[:,:3],b[:,:3],atol=0,rtol=0)
