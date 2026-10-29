import pytest
import torch
import torch.nn.functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import ProbabilityWorkspace
from compute_trainium.budget_conditioned_workspace import BudgetConditionedWorkspace
from compute_trainium.probability_intervention import ProposalReadout
from compute_trainium.probability_model_factory import probability_model

def config():return ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=12)

def pair():
    torch.manual_seed(731);old=ProbabilityWorkspace(config()).double();rng=torch.get_rng_state()
    torch.manual_seed(731);new=BudgetConditionedWorkspace(config()).double()
    assert torch.equal(rng,torch.get_rng_state())
    return old,new

@pytest.mark.parametrize('k',[0,1,2])
def test_zero_cues_preserve_initial_function_and_shared_gradients(k):
    old,new=pair();x=torch.randint(23,(2,7));y=torch.randint(23,(2,7))
    torch.testing.assert_close(old(x,k),new(x,k),rtol=0,atol=0)
    for m in (old,new):F.cross_entropy(m(x,k).flatten(0,1),y.flatten()).backward()
    # Extra additions change backward accumulation order at FP64 roundoff only.
    # Zero-budget identity remains bit-exact, including its optimizer update.
    for name,parameter in old.named_parameters():
        torch.testing.assert_close(parameter.grad,dict(new.named_parameters())[name].grad,
                                   rtol=0 if k==0 else 1e-12,atol=0 if k==0 else 1e-14)
    assert new.remaining_budget_cues.grad is not None
    assert bool(new.remaining_budget_cues.grad.abs().sum()>0)==(k>0)

def test_zero_budget_remains_exact_with_learned_nonzero_cues_and_update():
    old,new=pair();new.remaining_budget_cues.data.normal_();x=torch.randint(23,(2,7));y=torch.randint(23,(2,7))
    opts=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (old,new)]
    for m,opt in zip((old,new),opts):F.cross_entropy(m(x,0).flatten(0,1),y.flatten()).backward();opt.step()
    for name,parameter in old.named_parameters():assert torch.equal(parameter,dict(new.named_parameters())[name])
    assert torch.count_nonzero(new.remaining_budget_cues.grad)==0

def test_conditioned_content_intervention_is_causal_and_uses_cues():
    _,m=pair();m.remaining_budget_cues.data.normal_(std=.1);x=torch.randint(23,(2,8));changed=x.clone();changed[:,5:]=(changed[:,5:]+1)%23
    identity=ProposalReadout(m,2,permutation=torch.arange(23))
    torch.testing.assert_close(identity(x),m(x,2),rtol=0,atol=0)
    perm=torch.randperm(23);reader=ProposalReadout(m,2,permutation=perm)
    torch.testing.assert_close(reader(x)[:,:5],reader(changed)[:,:5],rtol=0,atol=0)
    assert not torch.equal(reader(x),m(x,2))

def test_factory_old_plan_and_independent_explicit_attention():
    p={'model_config':config().__dict__};torch.manual_seed(4);a=probability_model(p);torch.manual_seed(4);b=ProbabilityWorkspace(config())
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)
    p['remaining_budget_conditioning']=True;a=probability_model(p).double();b=probability_model(p,explicit_attention=True).double();b.load_state_dict(a.state_dict())
    a.remaining_budget_cues.data.normal_(std=.1);b.load_state_dict(a.state_dict());x=torch.randint(23,(2,7));y=torch.randint(23,(2,7))
    for k in (0,1,2):
        a.zero_grad();b.zero_grad()
        for m in (a,b):F.cross_entropy(m(x,k).flatten(0,1),y.flatten()).backward()
        for u,v in zip(a.parameters(),b.parameters()):torch.testing.assert_close(u.grad,v.grad,rtol=1e-9,atol=1e-11)


def test_cue_has_own_gradient_family_and_initial_shared_digest():
    from compute_trainium.probability_model_factory import gradient_families
    from scripts.qualify_native import digest
    old,new=pair();state=new.state_dict()
    shared=type(state)((n,v) for n,v in state.items() if n!='remaining_budget_cues')
    assert digest(shared)==digest(old.state_dict())
    families=gradient_families({'remaining_budget_conditioning':True})
    assert dict(families)['budget_cues']('remaining_budget_cues')
    for name,_ in new.named_parameters():assert sum(match(name) for _,match in families)==1
