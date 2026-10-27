import pytest
import torch
from compute_trainium.stable_target_loss import stable_target_log_probs


@pytest.mark.parametrize('margin',[-100.,0.,15.,30.,50.])
def test_target_loss_and_gradient_match_independent_fp64_probability_oracle(margin):
    torch.manual_seed(7523)
    logits=torch.randn(3,257,dtype=torch.float32)*.1;targets=torch.tensor([0,125,256])
    logits[torch.arange(3),targets]=margin;logits.requires_grad_(True)
    loss=-stable_target_log_probs(logits,targets).sum();loss.backward()
    # Independent analytic double-precision gradient;never compute p_y-1.
    probability=torch.softmax(logits.detach().double(),-1);expected=probability.clone()
    expected[torch.arange(3),targets]=0.
    off_sum=expected.sum(-1)
    expected[torch.arange(3),targets]=-off_sum
    torch.testing.assert_close(logits.grad.double(),expected,rtol=3e-5,atol=1e-30)
    if margin>=30:assert bool((logits.grad[torch.arange(3),targets]<0).all())
    if abs(margin)<20:
        reference=torch.nn.functional.cross_entropy(logits.detach().double(),targets,reduction='sum')
        torch.testing.assert_close(loss.double(),reference,rtol=3e-6,atol=1e-6)
