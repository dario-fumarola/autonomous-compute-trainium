import copy
import torch
import torch.nn.functional as F
from compute_trainium.recoverable_dependencies import pack_dependency,dependency_metrics
from compute_trainium.workspace_olmo_candidate import OlmoWorkspaceConfig,rotary_tensors
from compute_trainium.sparse_workspace_candidate import IndependentSparseWorkspaceCandidate


def case():return dict(full_prefix=list(range(4,44)),short_prefix=list(range(36,44)),answer=[44,45,46],short_offset=32)


def test_actual_targets_and_rope_positions_are_identical_under_context_removal():
    row=case();a=pack_dependency(row,'full',63,positions=64);b=pack_dependency(row,'short',63,positions=64)
    assert torch.equal(a['targets'],b['targets']) and int(a['mask'].sum())==3
    assert torch.equal(a['positions'][a['anchors']],b['positions'][b['anchors']])
    assert torch.equal(a['tokens'][0,32:42],b['tokens'][0,:10])
    assert int(a['anchors'].max())<64 and int(b['anchors'].max())<64


def test_future_padding_and_omitted_workspace_queries_cannot_change_real_outputs():
    torch.manual_seed(5701);c=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63);m=IndependentSparseWorkspaceCandidate(c).eval();v=pack_dependency(case(),'full',63,positions=64);cos,sin=rotary_tensors(c,64)
    with torch.no_grad():
        first=m.sparse_forward(v['tokens'],cos,sin,v['anchors'],raw_positions=v['anchors']);other=v['tokens'].clone();other[:,42:]=7
        changed=m.sparse_forward(other,cos,sin,v['anchors'][:3],raw_positions=v['anchors'][:3])
    torch.testing.assert_close(first['raw_logits'][:,:3],changed['raw_logits'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(first['mixed_logits'][:,:3],changed['mixed_logits'],rtol=1e-5,atol=1e-6)


def test_metrics_match_independent_masked_cross_entropy_and_ignore_fake_targets():
    torch.manual_seed(5702);raw=torch.randn(1,16,64,dtype=torch.float64);work=torch.randn_like(raw);v=pack_dependency(case(),'full',63,positions=64);x=torch.randn(1,16,8,dtype=torch.float64);y=torch.randn_like(x)
    got=dependency_metrics(raw,work,v['targets'],v['mask'],x,y)
    expected=F.cross_entropy(raw[:,:3].reshape(-1,64),v['targets'][:,:3].reshape(-1));torch.testing.assert_close(got['raw_nll'],expected)
    assert float(got['real_targets'])==3 and 0<=float(got['harmful_flip_rate'])<=1 and float(got['raw_to_workspace_kl'])>=0
    edited=raw.clone();edited[:,3:]=1e6;changed=dependency_metrics(edited,work,v['targets'],v['mask'],x,y);torch.testing.assert_close(got['raw_nll'],changed['raw_nll'])


def test_rotary_global_shift_preserves_raw_and_workspace_predictions():
    torch.manual_seed(5703);c=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63);m=IndependentSparseWorkspaceCandidate(c).eval();v=pack_dependency(case(),'full',63,positions=64);cos,sin=rotary_tensors(c,96)
    with torch.no_grad():
        a=m.sparse_forward(v['tokens'],cos[:,:,:64],sin[:,:,:64],v['anchors'],raw_positions=v['anchors']);b=m.sparse_forward(v['tokens'],cos[:,:,32:96],sin[:,:,32:96],v['anchors'],raw_positions=v['anchors'])
    for key in ['raw_logits','mixed_logits']:torch.testing.assert_close(a[key],b[key],rtol=2e-5,atol=2e-6)


def test_original_variant_is_exact_and_strict_embedding_replay_is_identity():
    from compute_trainium.dependency_candidate import DependencyCandidate
    torch.manual_seed(5711);c=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63);old=IndependentSparseWorkspaceCandidate(c);model=DependencyCandidate(c);model.load_state_dict(old.state_dict());v=pack_dependency(case(),'full',63,positions=64);cos,sin=rotary_tensors(c,64)
    a=old.sparse_forward(v['tokens'],cos,sin,v['anchors'],raw_positions=v['anchors']);b=model.dependency_forward(v['tokens'],cos,sin,v['anchors'],raw_positions=v['anchors']);assert torch.equal(a['raw_logits'],b['raw_logits']) and torch.equal(a['mixed_logits'],b['mixed_logits'])
    original=copy.deepcopy(model);replayed=copy.deepcopy(model);raw=original(v['tokens'],cos,sin,positions=v['anchors']);replay=replayed.dependency_forward(v['tokens'],cos,sin,v['anchors'],raw_positions=v['anchors'],variant='embedding_replay')['mixed_logits'];torch.testing.assert_close(raw,replay,rtol=2e-5,atol=2e-6)
    raw.square().mean().backward();replay.square().mean().backward()
    for p,q in zip(original.parameters(),replayed.parameters(),strict=True):torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)


def test_rms_control_matches_source_embedding_scale_without_new_parameters():
    from compute_trainium.dependency_candidate import DependencyCandidate
    torch.manual_seed(5712);c=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63);m=DependencyCandidate(c);v=pack_dependency(case(),'full',63,positions=64);cos,sin=rotary_tensors(c,64)
    r=m.dependency_forward(v['tokens'],cos,sin,v['anchors'],raw_positions=v['anchors'],variant='strict_rms');a=r['workspace_input'].square().mean(-1).sqrt();b=m.model.embed_tokens(v['tokens']).index_select(1,v['anchors']).square().mean(-1).sqrt();torch.testing.assert_close(a,b)
    assert list(m.state_dict())==list(IndependentSparseWorkspaceCandidate(c).state_dict())


def test_trimming_fully_masked_future_cache_preserves_outputs_and_all_gradients():
    from compute_trainium.dependency_candidate import DependencyCandidate
    torch.manual_seed(5713);c=OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63);m=DependencyCandidate(c);v=pack_dependency(case(),'full',63,positions=64);cos,sin=rotary_tensors(c,64)
    for variant in ['original','strict','strict_rms','embedding_replay']:
        a=copy.deepcopy(m);b=copy.deepcopy(m);left=a.dependency_forward(v['tokens'],cos,sin,v['anchors'],raw_positions=v['anchors'],variant=variant);right=b.dependency_forward(v['tokens'],cos,sin,v['anchors'],raw_positions=v['anchors'],variant=variant,cache_prefix_length=56)
        for k in ['raw_logits','mixed_logits']:torch.testing.assert_close(left[k],right[k],rtol=2e-5,atol=2e-6)
        left['mixed_logits'].square().mean().backward();right['mixed_logits'].square().mean().backward()
        for p,q in zip(a.parameters(),b.parameters(),strict=True):torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)
