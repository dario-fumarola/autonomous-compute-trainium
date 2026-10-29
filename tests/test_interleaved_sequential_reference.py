import copy
import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_sequential_reference import cached_sequential


@pytest.mark.parametrize('kv_heads', [1, 2])
def test_cached_reference_matches_recomputed_states_scores_and_full_gradients(kv_heads):
    torch.manual_seed(8201)
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=kv_heads,
                    ffn_dim=32, vocab_size=23, max_seq=8)
    cached = InterleavedWorkspace(c).double()
    recomputed = copy.deepcopy(cached)
    x, y = torch.randint(23, (2, 5)), torch.randint(23, (2, 5))
    a, za = cached_sequential(cached, x, return_states=True)
    b, zb = recomputed.sequential(x, return_states=True)
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(za, zb, atol=1e-10, rtol=1e-10)
    F.cross_entropy(a.flatten(0, 1), y.flatten()).backward()
    F.cross_entropy(b.flatten(0, 1), y.flatten()).backward()
    for (name, p), (other, q) in zip(cached.named_parameters(), recomputed.named_parameters()):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-8)
    changed = x.clone(); changed[:, 3:] = (changed[:, 3:]+1) % 23
    torch.testing.assert_close(a[:, :3], cached_sequential(cached, changed)[:, :3], atol=0, rtol=0)


def test_cached_reference_is_not_an_accelerator_path():
    c = ModelConfig(dim=16, layers=1, heads=2, kv_heads=1, ffn_dim=32, vocab_size=23, max_seq=4)
    with pytest.raises(ValueError, match='CPU SDPA'):
        cached_sequential(InterleavedWorkspace(c, explicit_attention=True), torch.ones(1, 2, dtype=torch.long))
