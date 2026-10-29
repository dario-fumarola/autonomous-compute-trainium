import pytest
import torch
import torch.nn.functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace
from compute_trainium.probability_role_gradients import role_replay

@pytest.mark.parametrize('steps',[0,1,2])
def test_identical_forward_and_exact_embedding_gradient_decomposition(steps):
    torch.manual_seed(617)
    m=ProbabilityWorkspace(ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=12)).double()
    x=torch.randint(23,(2,7));y=torch.randint(23,(2,7))
    leaves=[m.embedding.weight.detach().clone().requires_grad_() for _ in range(3)]
    expected=m(x,steps);actual=role_replay(m,x,*leaves,steps)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    original=torch.autograd.grad(F.cross_entropy(expected.flatten(0,1),y.flatten()),m.embedding.weight)[0]
    parts=torch.autograd.grad(F.cross_entropy(actual.flatten(0,1),y.flatten()),leaves,allow_unused=True)
    total=sum(g for g in parts if g is not None)
    torch.testing.assert_close(total,original,rtol=1e-10,atol=1e-12)
    assert (parts[2] is None)==(steps==0)
