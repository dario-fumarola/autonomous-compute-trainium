import torch
from compute_trainium.workspace_learning import WorkspaceFormat
from compute_trainium.workspace_readout_diagnostic import readout_context
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
from compute_trainium.workspace_native_objective import SelectiveWorkspacePredictor


def test_tail_replays_prefix_only_and_empty_workspace_is_exact_raw_input():
    c=torch.arange(32).reshape(1,32);z=torch.tensor([[40,41],[42,43]]);fmt=WorkspaceFormat((50,),(51,))
    result=readout_context(c,z,fmt,tail=True)
    assert torch.equal(result,torch.cat([c.expand(2,-1),z,c[:,-16:].expand(2,-1)],1))
    assert torch.equal(readout_context(c,z[:,:0],fmt,tail=True),c.expand(2,-1))
    assert torch.equal(readout_context(c,z,fmt,tail=False),torch.cat([z,c.expand(2,-1)],1))


def test_greedy_writer_matches_independent_unpadded_forward_and_preserves_rng():
    torch.manual_seed(8191)
    cfg=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63)
    model=WorkspaceOlmoCandidate(cfg).eval();actor=SelectiveWorkspacePredictor(model,positions=16)
    c=torch.tensor([[1,2,3,4]]);fmt=WorkspaceFormat((20,),());rng=torch.Generator().manual_seed(8193);state=rng.get_state()
    actual=actor.sample(c,format=fmt,length=4,group_size=1,generator=rng,greedy=True)
    context=torch.cat([c,torch.tensor([[20]])],1);expected=[]
    with torch.no_grad():
        for step in range(4):
            token=model(context,*rotary_tensors(cfg,context.shape[1]))[:,-1].argmax(-1,keepdim=True)
            expected.append(token);context=torch.cat([context,token],1)
    assert torch.equal(actual,torch.cat(expected,1)) and torch.equal(state,rng.get_state())
