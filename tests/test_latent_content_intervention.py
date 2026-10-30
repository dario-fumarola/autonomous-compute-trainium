"""Random numerical arrays qualify software, never research examples."""
import copy
import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_sequential_reference import cached_sequential
from compute_trainium.ephemeral_workspace import EphemeralWorkspace
from compute_trainium.latent_content_intervention import (
    MODES, _allowed, cached_intervention, recomputed_intervention)


def setup():
    torch.manual_seed(730283)
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=1,
                    ffn_dim=32, vocab_size=23, max_seq=8)
    return InterleavedWorkspace(c).double(), torch.randint(23, (2, 5)), torch.randint(23, (2, 5))


@pytest.mark.parametrize('mode', MODES)
def test_cached_full_prefix_outputs_states_and_gradients(mode):
    m, x, y = setup(); other = copy.deepcopy(m)
    a = cached_intervention(m, x, mode=mode, return_states=True)
    b = recomputed_intervention(other, x, mode=mode, return_states=True)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)
    F.cross_entropy(a[0].flatten(0, 1), y.flatten()).backward()
    F.cross_entropy(b[0].flatten(0, 1), y.flatten()).backward()
    for (name, p), (other_name, q) in zip(m.named_parameters(), other.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize('mode', MODES)
def test_selected_batch_and_future_invariance(mode):
    m, x, _ = setup()
    whole = cached_intervention(m, x, mode=mode)
    selected = cached_intervention(m, x, mode=mode, positions=[1, 4])
    torch.testing.assert_close(whole[:, [1, 4]], selected, atol=1e-12, rtol=1e-11)
    individual = torch.cat([cached_intervention(m, row[None], mode=mode) for row in x])
    torch.testing.assert_close(whole, individual, atol=1e-11, rtol=1e-10)
    changed = x.clone(); changed[:, 3:] = (changed[:, 3:] + 1) % 23
    torch.testing.assert_close(whole[:, :3], cached_intervention(m, changed, mode=mode)[:, :3], atol=0, rtol=0)


def test_authentic_current_only_and_constant_match_independent_existing_paths():
    m, x, _ = setup()
    torch.testing.assert_close(cached_intervention(m, x), cached_sequential(m, x), atol=1e-12, rtol=1e-11)
    ephemeral = EphemeralWorkspace(m.config).double(); ephemeral.load_state_dict(m.state_dict())
    torch.testing.assert_close(cached_intervention(m, x, mode='current_only'), ephemeral(x), atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(cached_intervention(m, x, mode='constant'), m.jacobi(x, mode='pause'), atol=1e-11, rtol=1e-10)


def test_lagged_input_uses_previous_generated_state_and_recomputes_future_states():
    m, x, _ = setup()
    _, genuine, _ = cached_intervention(m, x, return_states=True)
    _, generated, used = cached_intervention(m, x, mode='lagged', return_states=True)
    torch.testing.assert_close(used[:, :1], generated[:, :1], atol=0, rtol=0)
    torch.testing.assert_close(used[:, 1:], generated[:, :-1], atol=0, rtol=0)
    assert not torch.allclose(genuine[:, 2:], generated[:, 2:], atol=1e-10, rtol=1e-10)


def test_current_only_mask_blocks_all_historical_latent_routes():
    a = _allowed(10, 'current_only', last_only=False, device='cpu')
    for q in range(10):
        for k in range(10):
            assert a[q, k].item() == (k <= q and (k % 2 == 0 or k == q))


@pytest.mark.parametrize('positions', [[], [1, 1], [3, 1], [-1], [5], [True]])
def test_invalid_readouts_rejected(positions):
    m, x, _ = setup()
    with pytest.raises(ValueError, match='positions'):
        cached_intervention(m, x, positions=positions)
