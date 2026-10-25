"""Causality of auxiliary random inputs; numerical software fixtures only."""
import pytest
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_training_candidate import initial_state
from compute_trainium.prefix_noise_policy import prefix_window_state_key


def test_state_key_ignores_document_identity_and_rejects_invalid_positions():
    assert prefix_window_state_key('different future A', 0) == 1_000_000_007
    assert prefix_window_state_key('different future B', 0) == 1_000_000_007
    assert prefix_window_state_key(None, 1) == 1_000_001_031
    for value in [-1, 1.5, True, 2**63]:
        with pytest.raises(ValueError):
            prefix_window_state_key('unused', value)


@pytest.mark.parametrize('depth', [1, 2])
def test_future_tokens_and_document_length_do_not_change_prefix_states_or_logits(depth):
    torch.manual_seed(1709)
    config = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=64, max_seq=16)
    model = ParcaeCandidate(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter)*.01)
        a = initial_state(config, prefix_window_state_key('document A', 0), 3240)
        b = initial_state(config, prefix_window_state_key('document B', 0), 3240)
        assert torch.equal(a, b)
        tokens = torch.arange(8)[None]
        changed = tokens.clone()
        changed[:, 4:] += 8
        left = model(tokens, a[:, :8], total_steps=depth, backward_cap=1)
        right = model(changed, b[:, :8], total_steps=depth, backward_cap=1)
        short = model(tokens[:, :4], a[:, :4], total_steps=depth, backward_cap=1)
        torch.testing.assert_close(left[:, :4], right[:, :4], atol=0, rtol=0)
        torch.testing.assert_close(left[:, :4], short, atol=2e-6, rtol=2e-6)
