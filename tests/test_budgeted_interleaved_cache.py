import pytest
import torch
from torch.nn import functional as F

from compute_trainium.model import ModelConfig
from compute_trainium.budgeted_interleaved_reference import BudgetedInterleavedReference
from compute_trainium.budgeted_interleaved_cache import cached_budget
from compute_trainium.interleaved_sequential_reference import cached_sequential


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_exact_cached_outputs_states_and_gradients_match_recomputation(budget):
    torch.manual_seed(40359)
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32, vocab_size=23, max_seq=5)
    m = BudgetedInterleavedReference(c).double()
    x, y = torch.randint(0, 23, (2, 5)), torch.randint(0, 23, (2, 5))
    a, states = cached_budget(m, x, budget, return_states=True)
    b, reference = m.sequential_reference(x, budget, return_states=True)
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    for s, t in zip(states, reference):
        torch.testing.assert_close(s, t, atol=1e-11, rtol=1e-10)
    F.cross_entropy(a.flatten(0, 1), y.flatten()).backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    F.cross_entropy(b.flatten(0, 1), y.flatten()).backward()
    for n, p in m.named_parameters():
        torch.testing.assert_close(grads[n], p.grad, atol=1e-10, rtol=1e-9)
    changed = x.clone(); changed[:, 3:] = (changed[:, 3:]+1) % 23
    torch.testing.assert_close(a[:, :3], cached_budget(m, changed, budget)[:, :3], atol=0, rtol=0)
    if budget == 1:
        torch.testing.assert_close(a, cached_sequential(m, x), atol=0, rtol=0)
