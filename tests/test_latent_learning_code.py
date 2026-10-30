import pytest
import torch
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.tied_interleaved_ordinary import TiedOrdinary
from scripts.readout_latent_learning_code import last_logits


@pytest.mark.parametrize('role',['ordinary','pause','latent'])
@pytest.mark.parametrize('kv_heads',[1,2])
def test_last_projection_preserves_exact_full_prefix_and_batch(role,kv_heads):
    torch.manual_seed(77)
    c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=kv_heads,ffn_dim=24,vocab_size=32,max_seq=16)
    m=(TiedOrdinary(c) if role=='ordinary' else InterleavedWorkspace(c)).double().eval()
    x=torch.randint(0,32,(2,5))
    with torch.inference_mode():
        expected=(m.sequential(x) if role=='latent' else m(x,mode=role))[:,-1]
        actual=last_logits(m,x,role)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(actual[:1],last_logits(m,x[:1],role),rtol=1e-10,atol=1e-10)
