"""Numerical software fixtures only; no generated research examples."""
import math
from types import SimpleNamespace

import pytest
import torch

from compute_trainium.stable_recurrence_candidate import (
    DiagonalInjection, recurrent_tail, injection_matrix_training_flops)
from third_party.parcae_reference.injection import DiagonalInjection as UpstreamInjection


class ReferenceInit:
    def fn(self, name, layer):
        def initialize(tensor):
            with torch.no_grad():
                if name == 'ssm_A_log':
                    tensor.zero_()
                elif name == 'ssm_dt_bias':
                    delta = -math.log(math.sqrt(1/5))
                    tensor.fill_(delta+math.log(-math.expm1(-delta)))
                elif name == 'ssm_B_identity':
                    tensor.copy_(torch.eye(tensor.shape[0]))
                else:
                    raise AssertionError(name)
        return initialize


def test_pinned_upstream_value_and_all_gradients():
    torch.manual_seed(32)
    candidate = DiagonalInjection(5).double()
    upstream = UpstreamInjection(SimpleNamespace(
        recurrent_embedding_dimension=5, n_embd=5, init=ReferenceInit())).double()
    # Compare away from the identity initialization too.
    with torch.no_grad():
        for p in candidate.parameters():
            p.add_(torch.randn_like(p)*.2)
    upstream.load_state_dict(candidate.state_dict())
    h = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    e = torch.randn_like(h, requires_grad=True)
    h2, e2 = h.detach().clone().requires_grad_(), e.detach().clone().requires_grad_()
    y, ref = candidate(h, e), upstream(h2, e2)
    torch.testing.assert_close(y, ref, rtol=1e-12, atol=1e-12)
    y.square().sum().backward()
    ref.square().sum().backward()
    for a, b in [(h, h2), (e, e2), *zip(candidate.parameters(), upstream.parameters())]:
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)


def test_independent_finite_difference():
    torch.manual_seed(17)
    m = DiagonalInjection(3).double()
    h = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    e = torch.randn_like(h, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda h, e, a, d, b: torch.func.functional_call(
            m, {'A_log': a, 'dt_bias': d, 'B': b}, (h, e)),
        (h, e, m.A_log, m.dt_bias, m.B), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_truncation_keeps_forward_and_input_tail_gradient():
    weight = torch.tensor(.7, dtype=torch.double, requires_grad=True)
    source = torch.tensor(1.3, dtype=torch.double, requires_grad=True)
    state = torch.tensor(.2, dtype=torch.double, requires_grad=True)
    full = recurrent_tail(lambda h, e: weight*h+e, state, source, 5, 5)
    cut = recurrent_tail(lambda h, e: weight*h+e, state, source, 5, 2)
    torch.testing.assert_close(full, cut, rtol=0, atol=0)
    gw, ge, gh = torch.autograd.grad(cut, (weight, source, state), allow_unused=True)
    h3 = state.detach()
    for _ in range(3):
        h3 = weight.detach()*h3+source.detach()
    # Tail is w*(w*h3 + e) + e: a separate analytic reference.
    torch.testing.assert_close(gw, 2*weight.detach()*h3+source.detach())
    torch.testing.assert_close(ge, weight.detach()+1)
    assert gh is None


def test_matrix_cost_counts_forward_only_prefix():
    assert injection_matrix_training_flops(8, 5, 2) == 64*(2*5+4*2)
    assert injection_matrix_training_flops(8, 1, 8) == 64*6
    with pytest.raises(ValueError):
        recurrent_tail(lambda h, e: h+e, torch.zeros(1), torch.ones(1), 0, 1)
