import torch
import pytest
from transformers import Olmo2Config,Olmo2ForCausalLM
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors


@pytest.mark.parametrize('checkpointed',[False,True])
def test_fp32_candidate_matches_independent_hf_forward_every_gradient_and_selection(checkpointed):
    torch.manual_seed(7103)
    config=Olmo2Config(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
        num_key_value_heads=4,vocab_size=64,pad_token_id=63,max_position_embeddings=64,
        rope_parameters={'rope_type':'default','rope_theta':500000.})
    config._attn_implementation='eager';reference=Olmo2ForCausalLM(config).eval()
    candidate=WorkspaceOlmoCandidate(OlmoWorkspaceConfig.from_hf(config),activation_checkpointing=checkpointed).eval()
    assert set(candidate.state_dict())==set(reference.state_dict())
    candidate.load_state_dict(reference.state_dict());inputs=torch.tensor([[1,2,3,4,5,6,7,8]])
    cos,sin=rotary_tensors(candidate.config,inputs.shape[1]);actual=candidate(inputs,cos,sin)
    expected=reference(inputs,use_cache=False).logits
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    labels=torch.tensor([[2,3,4,5,6,7,8,9]])
    torch.nn.functional.cross_entropy(actual.reshape(-1,64),labels.reshape(-1)).backward()
    torch.nn.functional.cross_entropy(expected.reshape(-1,64),labels.reshape(-1)).backward()
    gradients=dict(reference.named_parameters())
    for name,p in candidate.named_parameters():
        torch.testing.assert_close(p.grad,gradients[name].grad,rtol=2e-4,atol=2e-6)
    positions=torch.tensor([1,4,7]);torch.testing.assert_close(candidate(inputs,cos,sin,positions),expected[:,positions,:],rtol=2e-5,atol=2e-6)
