import pytest
import torch
from compute_trainium.join_execution_probe import Fixture

@pytest.mark.parametrize('case',range(8))
def test_operator_reference_and_gradients(case):
    a=Fixture(case,small=True).double();b=Fixture(case,reference=True,small=True).double()
    b.load_state_dict(a.state_dict());x=a.input().double().requires_grad_();y=x.detach().clone().requires_grad_()
    p=a(x);q=b(y);torch.testing.assert_close(p,q,atol=1e-10,rtol=1e-10)
    p.square().mean().backward();q.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-10,rtol=1e-10)
    for v,w in zip(a.parameters(),b.parameters()):
        assert (v.grad is None)==(w.grad is None)
        if v.grad is not None:torch.testing.assert_close(v.grad,w.grad,atol=1e-10,rtol=1e-10)
