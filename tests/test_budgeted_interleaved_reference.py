"""Software fixtures only; no generated research examples."""
import pytest
import torch
from torch.nn import functional as F

from compute_trainium.budgeted_interleaved_reference import (
    BudgetedInterleavedReference, grouped_layout, matrix_work)
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.grouped_interleaved_workspace import GroupedInterleavedWorkspace
from compute_trainium.model import ModelConfig


def fixture(explicit=False):
    torch.manual_seed(40331)
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=1,
                    ffn_dim=32, vocab_size=23, max_seq=4)
    return BudgetedInterleavedReference(c, explicit_attention=explicit).double(), torch.randint(0, 23, (2, 4)), torch.randint(0, 23, (2, 4))


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_mask_matches_chronological_access(budget):
    positions, allowed = grouped_layout(4, budget)
    order = [(budget+1)*t+s for s in range(budget+1) for t in range(4)]
    assert positions.tolist() == list(range(4))*(budget+1)
    for q in range(len(order)):
        for k in range(len(order)):
            assert bool(allowed[q, k]) == (order[k] <= order[q])


@pytest.mark.parametrize('explicit', [False, True])
def test_existing_ordinary_and_one_latent_graphs_outputs_and_gradients_match(explicit):
    m, x, y = fixture(explicit)
    old = GroupedInterleavedWorkspace(m.config, explicit_attention=explicit).double()
    old.load_state_dict(m.state_dict())
    for budget, mode in [(0, 'ordinary'), (1, 'latent')]:
        m.zero_grad(); old.zero_grad()
        a, b = m(x, budget, 2), old(x, 2, mode)
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
        for z in (a, b):
            F.cross_entropy(z.flatten(0, 1), y.flatten()).backward()
        for (_, p), (_, q) in zip(m.named_parameters(), old.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_finite_acyclic_convergence_matches_exact_outputs_states_and_full_gradients(budget):
    m, x, y = fixture()
    # At most budget*n generated ancestors; no convergence assumption is used
    # to authorize a smaller finite training approximation.
    a, states = m(x, budget, budget*x.shape[1], return_states=True)
    b, exact = m.sequential_reference(x, budget, return_states=True)
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    for s, t in zip(states, exact):
        torch.testing.assert_close(s, t, atol=1e-11, rtol=1e-10)
    F.cross_entropy(a.flatten(0, 1), y.flatten()).backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    F.cross_entropy(b.flatten(0, 1), y.flatten()).backward()
    for n, p in m.named_parameters():
        torch.testing.assert_close(grads[n], p.grad, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_future_tokens_cannot_change_prefix_predictions(budget):
    m, x, _ = fixture()
    other = x.clone(); other[:, 2:] = (other[:, 2:]+1) % 23
    torch.testing.assert_close(m(x, budget, 2)[:, :2], m(other, budget, 2)[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(m.sequential_reference(x, budget)[:, :2],
                               m.sequential_reference(other, budget)[:, :2], atol=0, rtol=0)


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_dispatched_forward_and_backward_work(budget):
    from torch.utils._python_dispatch import TorchDispatchMode
    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.work = 0
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.mm.default:
                a, b = args; self.work += 2*a.shape[0]*a.shape[1]*b.shape[1]
            elif func is torch.ops.aten.bmm.default:
                a, b = args; self.work += 2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
            return func(*args, **(kwargs or {}))
    m, x, y = fixture(True)
    counter = Count()
    with counter:
        logits = m(x, budget, 2)
        forward = counter.work
        F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
    ledger = matrix_work(m.config, 4, budget, 2, batch=2)
    assert forward == ledger['forward_matmul_flops']
    assert counter.work == ledger['training_matmul_flops']


@pytest.mark.parametrize('budget', [-1, 3, True, 1.5])
def test_invalid_budget_rejected(budget):
    m, x, _ = fixture()
    with pytest.raises(ValueError):
        m(x, budget)
    with pytest.raises(ValueError):
        m.sequential_reference(x, budget)


def test_same_parameter_inventory_and_one_latent_sequential_reference():
    m, x, _ = fixture()
    old = InterleavedWorkspace(m.config).double()
    old.load_state_dict(m.state_dict())
    assert list(m.state_dict()) == list(old.state_dict())
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in old.parameters())
    torch.testing.assert_close(m.sequential_reference(x, 1), old.sequential(x), atol=0, rtol=0)
