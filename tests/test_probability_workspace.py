import copy
import torch
import torch.nn.functional as F

from compute_trainium.model import LanguageModel, ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace, expected_embedding, matrix_work


def config():
    return ModelConfig(dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32, vocab_size=23, max_seq=16)


def test_zero_step_exact_ordinary_gradients_and_adamw():
    torch.manual_seed(91)
    ordinary=LanguageModel(config()).double()
    candidate=ProbabilityWorkspace(config()).double()
    candidate.load_state_dict(ordinary.state_dict())
    x=torch.randint(23,(2,8)); y=torch.randint(23,(2,8))
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (ordinary,candidate)]
    assert torch.equal(ordinary(x),candidate(x,0))
    for m,o in zip((ordinary,candidate),optimizers):
        F.cross_entropy(m(x).flatten(0,1),y.flatten()).backward();o.step()
    for a,b in zip(ordinary.parameters(),candidate.parameters()):
        assert torch.equal(a.grad,b.grad)
        assert torch.equal(a,b)


def test_full_expectation_matches_explicit_sum_and_gradient():
    torch.manual_seed(8)
    logits=torch.randn(2,3,7,dtype=torch.double,requires_grad=True)
    weight=torch.randn(7,5,dtype=torch.double,requires_grad=True)
    actual=expected_embedding(logits,weight)
    ref=(logits.softmax(-1).unsqueeze(-1)*weight).sum(-2)
    torch.testing.assert_close(actual,ref)
    ga=torch.autograd.grad(actual.square().sum(),(logits,weight),retain_graph=True)
    gb=torch.autograd.grad(ref.square().sum(),(logits,weight))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b)
    assert min(float(g.norm()) for g in ga)>0


def test_every_replay_remains_causal():
    torch.manual_seed(42)
    m=ProbabilityWorkspace(config(),explicit_attention=True).double()
    x=torch.randint(23,(1,9)); changed=x.clone();changed[:,5:]=(changed[:,5:]+1)%23
    for steps in (0,1,2):
        torch.testing.assert_close(m(x,steps)[:,:5],m(changed,steps)[:,:5],rtol=0,atol=0)


def test_explicit_native_path_against_independent_sdpa_gradients():
    torch.manual_seed(82)
    a=ProbabilityWorkspace(config()).double()
    b=ProbabilityWorkspace(config(),explicit_attention=True).double();b.load_state_dict(a.state_dict())
    x=torch.randint(23,(2,7));y=torch.randint(23,(2,7))
    for steps,proposal in ((0,'expected'),(1,'expected'),(2,'expected'),(2,'uniform')):
        a.zero_grad();b.zero_grad()
        torch.testing.assert_close(a(x,steps,proposal),b(x,steps,proposal),rtol=1e-10,atol=1e-12)
        for m in (a,b):F.cross_entropy(m(x,steps,proposal).flatten(0,1),y.flatten()).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-9,atol=1e-11)
            assert torch.isfinite(pa.grad).all()


def test_reference_replay_is_additive_embedding_not_final_hidden_feedback():
    torch.manual_seed(7)
    m=ProbabilityWorkspace(config()).double();x=torch.randint(23,(2,7))
    e=m.embedding(x)
    for _ in range(2):e=e+(F.linear(m.body(e),m.embedding.weight).softmax(-1)@m.embedding.weight)
    torch.testing.assert_close(m(x,2),F.linear(m.body(e),m.embedding.weight),rtol=0,atol=0)


def test_uniform_control_and_work_accounting_are_explicitly_cheaper():
    m=ProbabilityWorkspace(config()).double();x=torch.randint(23,(2,7))
    ref=F.linear(m.body(m.embedding(x)+2*m.embedding.weight.mean(0)),m.embedding.weight)
    torch.testing.assert_close(m(x,2,'uniform'),ref)
    a=matrix_work(config(),7,0);b=matrix_work(config(),7,1);c=matrix_work(config(),7,2,'uniform')
    assert b['training_matmul_flops']>2*a['training_matmul_flops']
    assert a['training_matmul_flops']==c['training_matmul_flops']


def test_invalid_steps_rejected():
    import pytest
    m=ProbabilityWorkspace(config());x=torch.zeros((1,3),dtype=torch.long)
    for k in (-1,1.5,True):
        with pytest.raises(ValueError):m(x,k)
