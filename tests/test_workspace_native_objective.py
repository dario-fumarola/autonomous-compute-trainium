import inspect
import torch
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor
from compute_trainium.workspace_learning import Reader,WorkspaceFormat,target_log_probs,score_workspaces,identity_reader_context,sample_workspaces


def test_selective_prefix_only_sampling_scores_and_rng_match_full_cpu_reference():
    torch.manual_seed(7229)
    model=WorkspaceOlmoCandidate(OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,vocab_size=32,pad_token_id=31)).eval()
    predictor=SelectiveWorkspacePredictor(model,positions=16);c=torch.tensor([[1,2,3,4]]);y=torch.tensor([[5,6]])
    full=Reader(lambda x:model(x,*rotary_tensors(model.config,x.shape[1])))
    fmt=WorkspaceFormat((7,),(8,));rng=torch.Generator().manual_seed(7231)
    state=rng.get_state();z=predictor.sample(c,format=fmt,length=2,group_size=4,generator=rng)
    rng.set_state(state);again=predictor.sample(c,format=fmt,length=2,group_size=4,generator=rng)
    assert torch.equal(z,again) and 'targets' not in inspect.signature(predictor.sample).parameters
    actual=predictor.score(c,y,z,format=fmt)
    expected=score_workspaces(full,c,y,z,format=fmt,cost_per_token=.001,context_builder=identity_reader_context)
    torch.testing.assert_close(actual['rewards'],expected['rewards'],rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(predictor.log_probs(c,y,'ordinary'),target_log_probs(full,c,y,phase='ordinary'),rtol=2e-5,atol=2e-6)
    assert predictor.ledger()['phases']['workspace_sampling']==4
    rng.set_state(state)
    reference_samples=sample_workspaces(full,c,format=fmt,length=2,group_size=4,generator=rng)
    assert torch.equal(z,reference_samples)
    single=predictor.sample(c,format=fmt,length=2,group_size=1,generator=torch.Generator().manual_seed(7337))
    assert single.shape==(1,2)


def test_selective_runtime_coefficient_backward_matches_full_batched_gradient():
    from compute_trainium.workspace_learning import reader_loss,policy_loss
    torch.manual_seed(7333)
    model=WorkspaceOlmoCandidate(OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,vocab_size=32,pad_token_id=31)).eval()
    predictor=SelectiveWorkspacePredictor(model,positions=16);c=torch.tensor([[1,2,3,4]]);y=torch.tensor([[5,6]])
    z=torch.tensor([[7,8],[9,10],[11,12],[13,14]]);rewards=torch.tensor([-.2,.03,.07,.1]);fmt=WorkspaceFormat((15,),(16,))
    full=Reader(lambda x:model(x,*rotary_tensors(model.config,x.shape[1])))
    raw=-target_log_probs(full,c[:,:1],torch.cat([c[:,1:],y],1),phase='raw').mean()
    read=reader_loss(full,c,y,z,format=fmt,context_builder=identity_reader_context)
    pg=policy_loss(full,c,z,rewards,format=fmt);(.5*raw+.5*read+.1*pg).backward()
    expected={n:p.grad.clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
    class LocalAccumulator:
        def __init__(self):self.values={}
        def add_and_clear(self):
            for n,p in model.named_parameters():
                self.values[n]=self.values.get(n,0)+p.grad.detach().clone();p.grad=None
        def install(self):
            for n,p in model.named_parameters():p.grad=self.values[n]
    predictor.backward(c,y,z,rewards,format=fmt,accumulator=LocalAccumulator(),world=1,scale=1)
    for n,p in model.named_parameters():torch.testing.assert_close(p.grad,expected[n],rtol=2e-4,atol=2e-6)
