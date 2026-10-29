"""Numerical software fixtures only; no trained checkpoints or research data."""
import pytest
import torch

from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace
from compute_trainium.chunked_streaming_reference import ChunkedStreamingReference
from compute_trainium.model import ModelConfig


def model(explicit=False):
    torch.manual_seed(147)
    return ChunkedPersistentWorkspace(ModelConfig(dim=16,layers=2,heads=4,kv_heads=2,
        ffn_dim=24,vocab_size=31,max_seq=16),explicit_attention=explicit).double().eval()


@pytest.mark.parametrize('family', ['ordinary', 'generated', 'constant'])
@pytest.mark.parametrize('explicit', [False,True])
def test_single_token_streaming_agrees_at_every_prefix(family,explicit):
    m=model(explicit);stream=ChunkedStreamingReference(m,family,stride=4)
    tokens=torch.tensor([[2,5,8,11,14,17,20,23],[3,7,9,13,15,19,21,25]])
    steps=int(family!='ordinary');mode='pause' if family=='constant' else 'latent'
    for i in range(tokens.shape[1]):
        actual=stream.append(tokens[:,i:i+1])
        with torch.inference_mode():expected=m.full_reference(tokens[:,:i+1],4,steps,mode)[:,-1:]
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
        assert not actual.requires_grad
        inv=stream.cache_inventory();stored=i+1+steps*((i+1)//4)
        assert inv['stored_positions_per_layer']==stored
        assert inv['kv_tensor_bytes']==2*2*2*stored*2*4*8


@pytest.mark.parametrize('family', ['ordinary','generated','constant'])
def test_arbitrary_prefill_and_decode_chunks_preserve_global_boundaries(family):
    m=model();tokens=(torch.arange(32).view(2,16)%31).long()
    streaming=ChunkedStreamingReference(m,family,stride=4)
    outputs=[];left=0
    for count in (3,1,5,4,3):
        outputs.append(streaming.append(tokens[:,left:left+count]));left+=count
    with torch.inference_mode():
        expected=m.cached(tokens,4,int(family!='ordinary'),'pause' if family=='constant' else 'latent')
    torch.testing.assert_close(torch.cat(outputs,1),expected,atol=1e-12,rtol=1e-10)
    assert streaming.source_positions==16


def test_rejects_stale_cache_and_invalid_inputs_before_mutation():
    m=model();s=ChunkedStreamingReference(m,'generated',stride=4)
    s.append(torch.tensor([[1,2,3]]));before=s.cache_inventory()
    for bad in (torch.empty((1,0),dtype=torch.long),torch.tensor([[31]]),
                torch.tensor([[1.]]),torch.tensor([[1],[2]]),torch.ones(1,14,dtype=torch.long)):
        with pytest.raises(ValueError):s.append(bad)
        assert s.cache_inventory()==before
    with torch.no_grad():m.embedding.weight.add_(.001)
    with pytest.raises(RuntimeError,match='Weights'):s.append(torch.tensor([[4]]))


def test_new_session_does_not_share_state_and_failure_invalidates_cache(monkeypatch):
    m=model();first=ChunkedStreamingReference(m,'generated',stride=4)
    second=ChunkedStreamingReference(m,'generated',stride=4)
    first.append(torch.tensor([[1,2,3]]))
    assert second.cache_inventory()['stored_positions_per_layer']==0
    original=m.embeddings
    def fails(tokens):raise RuntimeError('numerical fixture failure')
    monkeypatch.setattr(m,'embeddings',fails)
    with pytest.raises(RuntimeError,match='fixture failure'):first.append(torch.tensor([[4]]))
    monkeypatch.setattr(m,'embeddings',original)
    with pytest.raises(RuntimeError,match='invalidates'):first.append(torch.tensor([[4]]))
    with pytest.raises(RuntimeError,match='Incomplete'):first.cache_inventory()
    assert second.append(torch.tensor([[1]])).shape==(1,1,31)


def test_training_mode_is_not_a_streaming_reference():
    m=model().train()
    with pytest.raises(ValueError):ChunkedStreamingReference(m,'generated')
    m.eval();s=ChunkedStreamingReference(m,'generated')
    m.train()
    with pytest.raises(RuntimeError):s.append(torch.tensor([[1]]))
