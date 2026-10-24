import pytest
import torch

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate


@pytest.mark.parametrize('recurrent', [False, True])
def test_disabled_mixed_policy_preserves_full_function_and_gradients(recurrent):
    torch.manual_seed(813)
    config = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=128, max_seq=8)
    original = ParcaeStagesCandidate(config, recurrent=recurrent, initialize=False)
    candidate = ParcaeFP16Candidate(config, recurrent=recurrent, fp16_body=False, initialize=False)
    candidate.load_state_dict(original.state_dict())
    tokens = torch.randint(0, 128, (1, 8))
    state = torch.randn(1, 8, 32) if recurrent else None
    kwargs = {'total_steps': 4, 'backward_cap': 2} if recurrent else {}
    left, right = original(tokens, state, **kwargs), candidate(tokens, state, **kwargs)
    assert torch.equal(left, right)
    left.square().mean().backward(); right.square().mean().backward()
    assert set(original.state_dict()) == set(candidate.state_dict())
    for a, b in zip(original.parameters(), candidate.parameters(), strict=True):
        assert torch.equal(a.grad, b.grad)


def test_fp16_stages_keep_head_masters_gradients_fp32_and_compile():
    torch.manual_seed(817)
    config = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=128, max_seq=8)
    eager = ParcaeFP16Candidate(config, initialize=False)
    compiled = ParcaeFP16Candidate(config, initialize=False)
    compiled.load_state_dict(eager.state_dict())
    compiled.compile_stages('aot_eager')
    tokens = torch.randint(0, 128, (1, 8))
    state = torch.randn(1, 8, 32)
    left, right = eager(tokens, state, total_steps=4, backward_cap=2), compiled(tokens, state, total_steps=4, backward_cap=2)
    assert left.dtype == right.dtype == torch.float32
    assert torch.equal(left, right)
    (left.square().mean()*8192).backward(); (right.square().mean()*8192).backward()
    for a, b in zip(eager.parameters(), compiled.parameters(), strict=True):
        assert a.dtype == b.dtype == a.grad.dtype == b.grad.dtype == torch.float32
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        assert torch.equal(a.grad, b.grad)
