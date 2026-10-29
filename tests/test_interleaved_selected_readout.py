import pytest
import torch
from torch.nn import functional as F

from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_selected_readout import cached_selected_logits
from compute_trainium.model import ModelConfig


@pytest.mark.parametrize('positions', [[0], [4], [0, 2, 4], [0, 1, 2, 3, 4]])
def test_selected_readout_matches_full_prefix_predictions_and_gradients(positions):
    torch.manual_seed(40292)
    c = ModelConfig(dim=8, layers=2, heads=2, kv_heads=1, ffn_dim=16, vocab_size=17, max_seq=5)
    m = InterleavedWorkspace(c).double()
    x = torch.randint(0, 17, (2, 5))
    y = torch.randint(0, 17, (2, len(positions)))
    full = m.sequential(x)[:, positions]
    selected = cached_selected_logits(m, x, positions)
    torch.testing.assert_close(full, selected, atol=1e-11, rtol=1e-10)
    F.cross_entropy(full.flatten(0, 1), y.flatten()).backward()
    grads = {n:p.grad.clone() for n,p in m.named_parameters()}
    m.zero_grad()
    F.cross_entropy(selected.flatten(0, 1), y.flatten()).backward()
    for n,p in m.named_parameters():
        torch.testing.assert_close(p.grad, grads[n], atol=1e-10, rtol=1e-9)
    changed = x.clone()
    changed[:, 3:] = (changed[:, 3:]+1) % 17
    a = cached_selected_logits(m, x, [0, 2])
    b = cached_selected_logits(m, changed, [0, 2])
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('positions', [[], [True], [-1], [5], [2, 1], [2, 2]])
def test_invalid_positions(positions):
    c = ModelConfig(dim=8, layers=1, heads=2, kv_heads=1, ffn_dim=16, vocab_size=17, max_seq=5)
    with pytest.raises(ValueError):
        cached_selected_logits(InterleavedWorkspace(c), torch.ones(1, 5, dtype=torch.long), positions)
