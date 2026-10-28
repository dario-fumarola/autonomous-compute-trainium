import copy
import pytest
import torch
import torch.nn.functional as F
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
from compute_trainium.sparse_workspace_candidate import IndependentSparseWorkspaceCandidate
from compute_trainium.spliced_workspace_training import partition_positions,sample_positions,spliced_objective,spliced_matrix_work
from compute_trainium.sparse_workspace_training import pack_view
from test_workspace_matrix_count import MatrixCounter


def config():return OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63)


def test_disabled_workspace_is_exact_in_outputs_gradients_and_adamw_step():
    torch.manual_seed(5500);base=WorkspaceOlmoCandidate(config(),activation_checkpointing=True)
    candidate=IndependentSparseWorkspaceCandidate(config(),activation_checkpointing=True);candidate.load_state_dict(base.state_dict())
    tokens=torch.arange(8).reshape(1,-1);targets=(tokens+1)%62;cos,sin=rotary_tensors(config(),8)
    left=base(tokens,cos,sin);right=candidate.sparse_forward(tokens,cos,sin,torch.tensor([2,5,7]),mix=0)['raw_logits']
    assert torch.equal(left,right)
    F.cross_entropy(left.reshape(-1,64),targets.reshape(-1)).backward();F.cross_entropy(right.reshape(-1,64),targets.reshape(-1)).backward()
    for p,q in zip(base.parameters(),candidate.parameters(),strict=True):assert torch.equal(p.grad,q.grad)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False) for m in [base,candidate]]
    for optimizer in optimizers:optimizer.step()
    for p,q in zip(base.parameters(),candidate.parameters(),strict=True):
        assert torch.equal(p,q)
        for field in ['exp_avg','exp_avg_sq','step']:assert torch.equal(optimizers[0].state[p][field],optimizers[1].state[q][field])


def test_sampled_partition_covers_every_natural_target_once_and_rng_replays():
    g=torch.Generator().manual_seed(5501);state=g.get_state();a,r=sample_positions(g)
    assert a.shape==(32,) and r.shape==(127,) and int(a.min())>=31
    assert torch.equal(torch.cat([a,r]).sort().values,torch.arange(159))
    g.set_state(state);aa,rr=sample_positions(g)
    assert torch.equal(a,aa) and torch.equal(r,rr)
    with pytest.raises(ValueError):partition_positions(torch.tensor([31,31]))


def test_removing_past_workspaces_cannot_change_a_later_query_or_its_gradient():
    torch.manual_seed(5511);full=IndependentSparseWorkspaceCandidate(config());sparse=copy.deepcopy(full)
    tokens=torch.arange(8).reshape(1,-1);cos,sin=rotary_tensors(config(),8)
    x=full.sparse_forward(tokens,cos,sin,torch.tensor([2,5,7]));y=sparse.sparse_forward(tokens,cos,sin,torch.tensor([7]))
    torch.testing.assert_close(x['mixed_logits'][:,-1],y['mixed_logits'][:,0],rtol=1e-5,atol=1e-6)
    x['mixed_logits'][:,-1].square().mean().backward();y['mixed_logits'][:,0].square().mean().backward()
    for p,q in zip(full.parameters(),sparse.parameters(),strict=True):torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)


@pytest.mark.parametrize('mode',['latent','constant','ordinary'])
def test_uniform_spliced_loss_and_actual_matrix_dispatch(mode):
    torch.manual_seed(5519);c=config();m=IndependentSparseWorkspaceCandidate(c,activation_checkpointing=True)
    values=torch.arange(160).reshape(1,-1)%62;tokens,targets=pack_view(dict(prefix=values[:,:128],continuation=values[:,128:]),63)
    anchors,raw=sample_positions(torch.Generator().manual_seed(19));cos,sin=rotary_tensors(c,160);counter=MatrixCounter()
    with counter:
        loss,_,_=spliced_objective(m,tokens,targets,cos,sin,anchors,raw,mode=mode,pause_id=62);loss.backward()
    assert counter.operations==spliced_matrix_work(c,query_count=0 if mode=='ordinary' else 32)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with torch.no_grad():
        if mode=='ordinary':all_logits=m(tokens,cos,sin,positions=torch.arange(159))
        else:
            result=m.sparse_forward(tokens,cos,sin,anchors,raw_positions=raw,mode=mode,pause_id=62)
            all_logits=torch.empty(1,159,c.vocab_size);all_logits[:,raw]=result['raw_logits'];all_logits[:,anchors]=result['mixed_logits']
        expected=F.cross_entropy(all_logits.double().reshape(-1,c.vocab_size),targets.reshape(-1))
    torch.testing.assert_close(loss.detach().double(),expected,rtol=2e-6,atol=2e-6)


def test_reused_plain_ntp_prefix_matches_legacy_packing_loss_and_all_gradients():
    """Compatibility required for extending the original ordinary931 state."""
    from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor
    from compute_trainium.workspace_joint_terms import padded_prediction_inputs
    torch.manual_seed(5571);c=config();legacy=WorkspaceOlmoCandidate(c,activation_checkpointing=True)
    current=IndependentSparseWorkspaceCandidate(c,activation_checkpointing=True);current.load_state_dict(legacy.state_dict())
    full=(torch.arange(160)%62).reshape(1,-1);context,labels=full[:,:1],full[:,1:]
    old_tokens,where=padded_prediction_inputs(context,labels,160,c.pad_token_id)
    tokens,targets=pack_view(dict(prefix=full[:,:128],continuation=full[:,128:]),c.pad_token_id)
    assert torch.equal(tokens,old_tokens) and torch.equal(targets,labels) and torch.equal(where,torch.arange(159))
    old_loss=-SelectiveWorkspacePredictor(legacy,positions=160).log_probs(context,labels,'ordinary_ntp').mean()
    selected,raw=sample_positions(torch.Generator().manual_seed(1));cos,sin=rotary_tensors(c,160)
    new_loss,_,_=spliced_objective(current,tokens,targets,cos,sin,selected,raw,mode='ordinary')
    assert torch.equal(old_loss,new_loss)
    old_loss.backward();new_loss.backward()
    for p,q in zip(legacy.parameters(),current.parameters(),strict=True):assert torch.equal(p.grad,q.grad)
