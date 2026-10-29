import copy
import pytest
import torch
import torch.nn.functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace,matrix_work
from compute_trainium.untied_probability_workspace import UntiedProbabilityWorkspace
from compute_trainium.probability_model_factory import probability_model,gradient_families
from compute_trainium.probability_intervention import ProposalReadout
from compute_trainium.pretraining_optimizer import parameter_groups


def config():return ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=12)

def pair():
    torch.manual_seed(892);a=ProbabilityWorkspace(config()).double();rng=torch.get_rng_state()
    torch.manual_seed(892);b=UntiedProbabilityWorkspace(config()).double()
    assert torch.equal(rng,torch.get_rng_state())
    assert b.prediction_weight.data_ptr()!=b.embedding.weight.data_ptr()
    return a,b

@pytest.mark.parametrize('depth,proposal',[(0,'expected'),(1,'expected'),(2,'expected'),(2,'uniform')])
def test_copied_table_preserves_forward_and_splits_tied_gradient(depth,proposal):
    a,b=pair();x=torch.randint(23,(2,7));y=torch.randint(23,(2,7))
    torch.testing.assert_close(a(x,depth,proposal),b(x,depth,proposal),rtol=0,atol=0)
    for m in (a,b):F.cross_entropy(m(x,depth,proposal).flatten(0,1),y.flatten()).backward()
    torch.testing.assert_close(a.embedding.weight.grad,b.embedding.weight.grad+b.prediction_weight.grad,rtol=1e-10,atol=1e-12)
    for n,p in a.named_parameters():
        if n!='embedding.weight':torch.testing.assert_close(p.grad,dict(b.named_parameters())[n].grad,rtol=1e-10,atol=1e-12)


def test_distinct_tables_independent_attention_causality_and_intervention():
    _,m=pair();m.prediction_weight.data.normal_(std=.07)
    other=UntiedProbabilityWorkspace(config(),explicit_attention=True).double();other.load_state_dict(m.state_dict())
    x=torch.randint(23,(2,9));y=torch.randint(23,(2,9));changed=x.clone();changed[:,6:]=(changed[:,6:]+5)%23
    for k in (0,1,2):
        m.zero_grad();other.zero_grad()
        for q in (m,other):F.cross_entropy(q(x,k).flatten(0,1),y.flatten()).backward()
        torch.testing.assert_close(m(x,k),other(x,k),rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(m(x,k)[:,:6],m(changed,k)[:,:6],rtol=0,atol=0)
        for p,q in zip(m.parameters(),other.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-9,atol=1e-11)
    identity=ProposalReadout(m,2,permutation=torch.arange(23))
    torch.testing.assert_close(identity(x),m(x,2),rtol=0,atol=0)
    assert not torch.equal(ProposalReadout(m,2,permutation=torch.randperm(23))(x),m(x,2))


def test_optimizer_exemption_and_exact_next_step_restoration():
    _,m=pair();x=torch.randint(23,(2,7));y=torch.randint(23,(2,7))
    def optimizer(q):return torch.optim.AdamW(parameter_groups(q,q.embedding.weight),lr=.001,betas=(.9,.95),foreach=False)
    opt=optimizer(m);groups={id(p):g['weight_decay'] for g in opt.param_groups for p in g['params']}
    assert groups[id(m.embedding.weight)]==groups[id(m.prediction_weight)]==0
    assert groups[id(m.blocks[0].q.weight)]==.1
    def step(q,o):
        o.zero_grad();F.cross_entropy(q(x,1).flatten(0,1),y.flatten()).backward();o.step()
    step(m,opt);n=UntiedProbabilityWorkspace(config()).double();n.load_state_dict(copy.deepcopy(m.state_dict()));no=optimizer(n);no.load_state_dict(copy.deepcopy(opt.state_dict()))
    step(m,opt);step(n,no)
    for a,b in zip(m.parameters(),n.parameters()):assert torch.equal(a,b)
    for a,b in zip(opt.state.values(),no.state.values()):
        for k in a:assert torch.equal(a[k],b[k])
    assert not torch.equal(m.embedding.weight,m.prediction_weight)


def test_factory_inventory_families_and_old_optimizer_unchanged():
    a,b=pair();p={'model_config':config().__dict__,'untied_prediction_table':True}
    assert isinstance(probability_model(p),UntiedProbabilityWorkspace)
    assert sum(x.numel() for x in b.parameters())==config().parameter_count()+config().vocab_size*config().dim
    families=gradient_families(p)
    for name,_ in b.named_parameters():assert sum(int(f(name)) for _,f in families)==1
    oldgroups=parameter_groups(a,a.embedding.weight)
    for group in oldgroups:
        for v in group['params']:assert group['weight_decay']==(0 if v is a.embedding.weight or v.ndim<2 else .1)
    assert type(probability_model({'model_config':config().__dict__})) is ProbabilityWorkspace
    with pytest.raises(ValueError):probability_model(dict(p,remaining_budget_conditioning=True))
    with pytest.raises(ValueError):probability_model(dict(p,untied_prediction_table=1))


def test_declared_matrix_work_matches_executed_shape_counts():
    _,m=pair();x=torch.randint(23,(2,7))
    # Count every dense linear and expected-embedding matrix product from the
    # model graph through dispatch; attention is separately included by config.
    from torch.utils._python_dispatch import TorchDispatchMode
    class Count(TorchDispatchMode):
        def __init__(self):super().__init__();self.flops=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func is torch.ops.aten.mm.default:
                a,b=args;self.flops+=2*a.shape[0]*a.shape[1]*b.shape[1]
            elif func is torch.ops.aten.bmm.default:
                a,b=args;self.flops+=2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
            return func(*args,**(kwargs or {}))
    # Explicit attention exposes the two full-square attention products.
    q=UntiedProbabilityWorkspace(config(),explicit_attention=True).double();q.load_state_dict(m.state_dict())
    for k in (0,1,2):
        c=Count()
        with c:q(x,k)
        assert c.flops==matrix_work(config(),7,k,batch=2)['forward_matmul_flops']
