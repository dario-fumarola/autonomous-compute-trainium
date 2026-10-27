import pytest
import torch
from compute_trainium.workspace_discovery import discovery_terms,step_matrix_work,matched_updates,forward_matrix_work,backward_terms
from compute_trainium.workspace_learning import WorkspaceFormat,Reader,target_log_probs
from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,WorkspaceOlmoCandidate,rotary_tensors
from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor


def test_draft_reads_own_samples_and_pause_states_follow_prefix():
    c=torch.tensor([[1,2,3,4]]);y=torch.arange(5,37).reshape(1,-1)
    z=torch.arange(32).reshape(4,8);f=WorkspaceFormat((40,),(41,))
    draft=list(discovery_terms('draft',c,y,z,None,f,pause_id=42))
    for i,row in enumerate(draft[1:5]):
        assert torch.equal(row[1],torch.cat([z[i:i+1],c],1))
        assert torch.equal(row[2],y[:,:4])
    assert torch.equal(draft[-1][1],torch.cat([c,torch.tensor([[40]])],1))
    assert torch.equal(draft[-1][2],y[:,:8])
    pause=list(discovery_terms('pause',c,y,None,None,f,pause_id=42))[-1]
    assert torch.equal(pause[1][:,:4],c) and (pause[1][:,4:]==42).all()
    raw=list(discovery_terms('ordinary',c,y,None,None,f,pause_id=42))
    assert len(raw)==1 and torch.equal(raw[0][2],torch.cat([c[:,1:],y],1))


def test_matrix_budget_includes_every_generation_scorer_recompute_and_head():
    c=OlmoWorkspaceConfig();steps=matched_updates(c)
    assert steps['workspace']==48 and steps['ordinary']*64<=65600
    budget=48*step_matrix_work(c,'workspace')
    for arm,count in steps.items():
        assert count*step_matrix_work(c,arm)>=budget
        assert (count-1)*step_matrix_work(c,arm)<budget
    assert steps['ordinary']>steps['pause']>steps['draft']>steps['workspace']
    a=forward_matrix_work(c,batch=1,positions=160,projected=159)
    assert step_matrix_work(c,'ordinary')==4*a['body']+3*a['head']
    assert forward_matrix_work(c,batch=4,positions=192,projected=1)['body']==4*forward_matrix_work(c,batch=1,positions=192,projected=1)['body']


@pytest.mark.parametrize('arm',['ordinary','pause','draft'])
def test_condition_gradient_matches_independent_full_hf_style_loss(arm):
    torch.manual_seed(7799)
    cfg=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63)
    model=WorkspaceOlmoCandidate(cfg).eval();predictor=SelectiveWorkspacePredictor(model,positions=32)
    c=torch.tensor([[1,2,3,4]]);y=torch.arange(5,21).reshape(1,-1);z=torch.arange(32).reshape(4,8);fmt=WorkspaceFormat((40,),(41,))
    full=Reader(lambda x:model(x,*rotary_tensors(cfg,x.shape[1])))
    raw=-target_log_probs(full,c[:,:1],torch.cat([c[:,1:],y if arm=='ordinary' else y[:,:4]],1),phase='raw').mean()
    if arm=='ordinary':loss=raw
    elif arm=='pause':
        ctx=torch.cat([c,c.new_full((1,8),42)],1)
        loss=.5*raw-.5*target_log_probs(full,ctx,y[:,:4],phase='pause').mean()
    else:
        read=-target_log_probs(full,torch.cat([z,c.expand(4,-1)],1),y[:,:4],phase='read').mean()
        write=-target_log_probs(full,torch.cat([c,c.new_tensor([[40]])],1),y[:,:8],phase='write').mean()
        loss=.5*raw+.5*read+.1*write
    loss.backward();expected={n:p.grad.clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
    class Accumulator:
        def __init__(self):self.values={}
        def add_and_clear(self):
            for n,p in model.named_parameters():self.values[n]=self.values.get(n,0)+p.grad.detach().clone();p.grad=None
        def install(self):
            for n,p in model.named_parameters():p.grad=self.values[n]
    backward_terms(predictor,discovery_terms(arm,c,y,z,None,fmt,pause_id=42),accumulator=Accumulator(),world=1,scale=1)
    for n,p in model.named_parameters():torch.testing.assert_close(p.grad,expected[n],rtol=2e-4,atol=2e-6)
