import itertools
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from compute_trainium.model import LanguageModel,ModelConfig
from compute_trainium.workspace_learning import (Reader,ForwardLedger,WorkspaceFormat,
    framed_context,target_log_probs,sample_workspaces,score_workspaces,group_advantages,
    policy_loss,reader_loss,make_frozen_scorer,update_ema)


@pytest.fixture
def model():
    torch.manual_seed(173)
    return LanguageModel(ModelConfig(dim=16,layers=1,heads=2,kv_heads=1,ffn_dim=24,vocab_size=17,max_seq=64)).eval()


def test_continuation_alignment_matches_independent_prefix_forwards(model):
    c=torch.tensor([[1,2,3]]);y=torch.tensor([[4,5,6]])
    actual=target_log_probs(Reader(model),c,y,phase='fixture')
    expected=[]
    for i in range(y.shape[1]):
        logits=model(torch.cat([c,y[:,:i]],dim=1))[:,-1,:]
        expected.append(F.log_softmax(logits.float(),-1)[0,y[0,i]])
    torch.testing.assert_close(actual,torch.stack(expected).reshape(1,-1))
    changed=y.clone();changed[0,2]=7
    earlier=target_log_probs(Reader(model),c,changed,phase='fixture')
    torch.testing.assert_close(actual[:,:2],earlier[:,:2],rtol=0,atol=0)


def test_writer_has_only_prefix_and_replays_rng_independently_of_targets(model):
    c=torch.tensor([[1,2,3]]);fmt=WorkspaceFormat((8,),(9,))
    def generate():return sample_workspaces(Reader(model),c,format=fmt,length=4,group_size=4,
                                          generator=torch.Generator().manual_seed(71),temperature=.7)
    z=generate();scorer=Reader(make_frozen_scorer(model))
    score_workspaces(scorer,c,torch.tensor([[4,5]]),z,format=fmt)
    score_workspaces(scorer,c,torch.tensor([[12,13]]),z,format=fmt)
    torch.testing.assert_close(generate(),z,rtol=0,atol=0)
    assert not z.requires_grad and z.shape==(4,4)
    empty=sample_workspaces(Reader(model),c,format=fmt,length=0,group_size=4,
                           generator=torch.Generator().manual_seed(71))
    assert empty.shape==(4,0)


def test_group_advantages_detach_and_remove_constant_baseline_and_fixed_cost():
    rewards=torch.tensor([1.,2.,4.,5.],requires_grad=True)
    a=group_advantages(rewards)
    assert not a.requires_grad
    torch.testing.assert_close(a,torch.tensor([-8/3,-4/3,4/3,8/3]))
    torch.testing.assert_close(group_advantages(rewards+37),a)
    torch.testing.assert_close(group_advantages(torch.ones(4)),torch.zeros(4))
    with pytest.raises(ValueError):group_advantages(torch.tensor([float('nan'),0.]))


class ConstantPolicy(nn.Module):
    def __init__(self):super().__init__();self.logits=nn.Parameter(torch.tensor([.2,-.3]))
    def forward(self,x):return self.logits.expand(x.shape[0],x.shape[1],-1)


def test_group_policy_estimator_matches_exact_expected_reward_gradient():
    actor=ConstantPolicy();prefix=torch.tensor([[0]]);rewards=torch.tensor([2.,-1.])
    probabilities=actor.logits.softmax(-1).detach();estimate=0
    for actions in itertools.product(range(2),repeat=2):
        z=torch.tensor(actions).reshape(2,1)
        weight=probabilities[actions[0]]*probabilities[actions[1]]
        estimate=estimate+weight*policy_loss(Reader(actor),prefix,z,rewards[list(actions)],format=WorkspaceFormat())
    observed=torch.autograd.grad(estimate,actor.logits)[0]
    exact=torch.autograd.grad(-(actor.logits.softmax(-1)*rewards).sum(),actor.logits)[0]
    torch.testing.assert_close(observed,exact)


def test_reward_is_detached_reader_learns_and_scorer_only_moves_on_ema(model):
    c=torch.tensor([[1,2,3]]);y=torch.tensor([[4,5]]);z=torch.tensor([[6,7],[8,9]])
    fmt=WorkspaceFormat((10,),(11,));scorer=make_frozen_scorer(model)
    original={n:p.clone() for n,p in scorer.named_parameters()}
    scores=score_workspaces(Reader(scorer),c,y,z,format=fmt)
    assert all(not v.requires_grad for v in scores.values())
    loss=reader_loss(Reader(model),c,y,z,format=fmt)+policy_loss(Reader(model),c,z,scores['rewards'],format=fmt)
    loss.backward()
    assert model.embedding.weight.grad[z[0,0]].abs().sum()>0
    assert all(p.grad is None for p in scorer.parameters())
    optimizer=torch.optim.SGD(model.parameters(),lr=.01);optimizer.step()
    for name,p in scorer.named_parameters():torch.testing.assert_close(p,original[name],rtol=0,atol=0)
    update_ema(scorer,model,decay=.5)
    for name,p in scorer.named_parameters():
        torch.testing.assert_close(p,.5*original[name]+.5*dict(model.named_parameters())[name])


def test_content_intervention_recomputes_all_states_and_ledger_counts_it(model):
    ledger=ForwardLedger();reader=Reader(make_frozen_scorer(model),ledger)
    c=torch.tensor([[1,2]]);y=torch.tensor([[3,4]]);fmt=WorkspaceFormat((8,),(9,))
    z=torch.tensor([[5,6],[7,8]])
    first=score_workspaces(reader,c,y,z,format=fmt)
    second=score_workspaces(reader,c,y,z.flip(1),format=fmt)
    assert not torch.equal(first['workspace_log_probability'],second['workspace_log_probability'])
    assert ledger.summary()['forward_calls']==6
    assert framed_context(c,z,fmt).tolist()==[[1,2,8,5,6,9],[1,2,8,7,8,9]]
    assert ledger.summary()['processed_positions']>0


def test_policy_temperature_matches_sample_distribution_and_rejects_nonfinite():
    actor=ConstantPolicy();c=torch.tensor([[0]]);y=torch.tensor([[1]])
    observed=target_log_probs(Reader(actor),c,y,phase='fixture',temperature=.5)
    torch.testing.assert_close(observed[0,0],(actor.logits/.5).log_softmax(-1)[1])
    with pytest.raises(ValueError):target_log_probs(Reader(actor),c,y,phase='fixture',temperature=float('nan'))
