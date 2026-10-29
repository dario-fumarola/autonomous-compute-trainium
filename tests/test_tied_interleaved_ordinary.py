import copy,math
import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace
from compute_trainium.pretraining_optimizer import parameter_groups
from compute_trainium.tied_interleaved_ordinary import TiedOrdinary,make_model


def test_tied_path_matches_independent_scaled_ordinary_gradients_and_update():
    torch.manual_seed(10401)
    c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=8)
    candidate=TiedOrdinary(c,explicit_attention=True).double()
    reference=ProbabilityWorkspace(c).double();reference.load_state_dict(candidate.state_dict())
    x,y=torch.randint(23,(2,5)),torch.randint(23,(2,5))
    a=candidate(x);b=F.linear(reference.body(math.sqrt(2.5*c.dim)*reference.embedding(x)),reference.embedding.weight)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    F.cross_entropy(a.flatten(0,1),y.flatten()).backward();F.cross_entropy(b.flatten(0,1),y.flatten()).backward()
    assert 'prediction_weight' not in candidate.state_dict()
    assert candidate.prediction_weight is candidate.embedding.weight
    assert sum(p.numel() for p in candidate.parameters())==c.parameter_count()
    for (_,p),(_,q) in zip(candidate.named_parameters(),reference.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)
    for m in (candidate,reference):
        groups=parameter_groups(m,m.embedding.weight,decay=.1)
        ids=[id(p) for g in groups for p in g['params']]
        assert len(ids)==len(set(ids))==len(list(m.parameters()))
        torch.optim.AdamW(groups,lr=.001,foreach=False).step()
    for p,q in zip(candidate.parameters(),reference.parameters()):torch.testing.assert_close(p,q,atol=1e-10,rtol=1e-8)
    with pytest.raises(ValueError):candidate(x,1,'latent')


def test_tied_factory_and_ceiling_are_explicit():
    small=ModelConfig(dim=16,layers=1,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=8)
    with pytest.raises(ValueError):make_model(dict(model_config=small.__dict__,untied_prediction_table=True))
    assert isinstance(make_model(dict(model_config=small.__dict__,untied_prediction_table=False)),TiedOrdinary)
    c=ModelConfig(dim=512,layers=11,heads=8,kv_heads=8,ffn_dim=1376,vocab_size=32000,max_seq=512)
    assert c.parameter_count()==51179008<51747328
    assert c.flops(512,512)['forward_matmul_flops']==58300825600
