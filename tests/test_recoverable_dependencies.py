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
