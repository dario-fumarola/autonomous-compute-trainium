import copy
import pytest
import torch

from compute_trainium.chunked_persistent_factory import make_model,matrix_work
from compute_trainium.model import ModelConfig


def plan():
    c=ModelConfig(dim=16,layers=2,heads=4,kv_heads=2,ffn_dim=24,vocab_size=31,max_seq=512)
    return dict(model_config=c.__dict__,workspace_mechanism='chunked_persistent_generated_state',
                untied_prediction_table=False,source_stride=128)


@pytest.mark.parametrize('graph',[0,1,2,3])
def test_native_cached_and_independent_cpu_paths(graph):
    p=plan();torch.manual_seed(642);native=make_model(p,explicit_attention=True).double()
    torch.manual_seed(642);cpu=make_model(p).double()
    x=torch.arange(129).remainder(30).add(1).unsqueeze(0)
    a,b=native(x,graph),cpu(x,graph)
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
    y=(x+1)%31
    for values in (a,b):torch.nn.functional.cross_entropy(values.flatten(0,1),y.flatten()).backward()
    for (name,v),(other,w) in zip(native.named_parameters(),cpu.named_parameters()):
        assert name==other
        torch.testing.assert_close(v.grad,w.grad,atol=1e-11,rtol=1e-9)
    count=matrix_work(native.config,129,graph)
    assert count['stored_positions']==129+int(graph in (2,3))


@pytest.mark.parametrize('field,value',[('untied_prediction_table',True),('source_stride',64),('workspace_mechanism','wrong')])
def test_unregistered_graph_configuration_rejected(field,value):
    p=plan();p[field]=value
    with pytest.raises(ValueError):make_model(p)


def test_graph_ids_are_not_implicit_budgets():
    m=make_model(plan())
    with pytest.raises(ValueError):m(torch.ones(1,2,dtype=torch.long),2,'latent')
    with pytest.raises(ValueError):matrix_work(m.config,129,True)
