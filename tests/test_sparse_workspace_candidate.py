import copy
import torch
from compute_trainium.workspace_olmo_candidate import WorkspaceOlmoCandidate,OlmoWorkspaceConfig,rotary_tensors
from compute_trainium.sparse_workspace_candidate import SparseWorkspaceCandidate,sparse_step_mask


def config():return OlmoWorkspaceConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64,pad_token_id=63)


def test_original_path_and_inventory_are_exactly_unchanged():
    torch.manual_seed(8831);original=WorkspaceOlmoCandidate(config())
    torch.manual_seed(8831);candidate=SparseWorkspaceCandidate(config())
    assert list(original.state_dict())==list(candidate.state_dict())
    for k,v in original.state_dict().items():assert torch.equal(v,candidate.state_dict()[k])
    tokens=torch.arange(8).reshape(1,-1);cos,sin=rotary_tensors(config(),8)
    expected=original(tokens,cos,sin);actual=candidate(tokens,cos,sin)
    assert torch.equal(expected,actual)
    a=torch.tensor([2,5,7]);result=candidate.sparse_forward(tokens,cos,sin,a,mix=0)
    torch.testing.assert_close(result['mixed_logits'],expected[:,a],rtol=2e-6,atol=2e-6)


def test_no_future_token_or_later_latent_can_change_an_earlier_prediction():
    torch.manual_seed(8833);model=SparseWorkspaceCandidate(config());tokens=torch.arange(8).reshape(1,-1);cos,sin=rotary_tensors(config(),8);a=torch.tensor([2,5,7])
    first=model.sparse_forward(tokens,cos,sin,a)
    changed=tokens.clone();changed[:,3:]=torch.tensor([[21,22,23,24,25]])
    second=model.sparse_forward(changed,cos,sin,a)
    torch.testing.assert_close(first['mixed_logits'][:,0],second['mixed_logits'][:,0],rtol=0,atol=0)
    assert not torch.equal(first['mixed_logits'][:,-1],second['mixed_logits'][:,-1])
    mask=sparse_step_mask(8,a)
    assert mask.shape==(3,11) and not mask[0,3:8].any() and not mask[0,9:].any()


def test_direct_credit_differs_from_detach_with_identical_forward_values():
    torch.manual_seed(8837);model=SparseWorkspaceCandidate(config());other=copy.deepcopy(model)
    tokens=torch.arange(8).reshape(1,-1);cos,sin=rotary_tensors(config(),8);a=torch.tensor([2,5,7])
    first=model.sparse_forward(tokens,cos,sin,a);first['workspace_input'].retain_grad()
    first['mixed_logits'].square().mean().backward()
    assert first['workspace_input'].grad is not None and first['workspace_input'].grad.norm()>0
    second=other.sparse_forward(tokens,cos,sin,a,mode='detached');second['mixed_logits'].square().mean().backward()
    assert torch.equal(first['mixed_logits'],second['mixed_logits'])
    assert any(not torch.equal(p.grad,q.grad) for p,q in zip(model.parameters(),other.parameters(),strict=True))


def test_checkpointed_cache_preserves_all_forward_and_gradient_values():
    torch.manual_seed(8841);plain=SparseWorkspaceCandidate(config());checked=copy.deepcopy(plain);checked.activation_checkpointing=True
    tokens=torch.arange(8).reshape(1,-1);cos,sin=rotary_tensors(config(),8);a=torch.tensor([2,5,7])
    outputs=[]
    for m in [plain,checked]:
        r=m.sparse_forward(tokens,cos,sin,a);loss=r['mixed_logits'].square().mean()+r['raw_logits'].square().mean();loss.backward();outputs.append(r)
    assert torch.equal(outputs[0]['mixed_logits'],outputs[1]['mixed_logits'])
    for p,q in zip(plain.parameters(),checked.parameters(),strict=True):torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)


def test_cached_graph_matches_independent_dense_two_dimensional_mask_and_gradients():
    import torch.nn.functional as F
    torch.manual_seed(8849);cached=SparseWorkspaceCandidate(config());dense=WorkspaceOlmoCandidate(config());dense.load_state_dict(cached.state_dict())
    tokens=torch.arange(8).reshape(1,-1);anchors=torch.tensor([2,5,7]);cos,sin=rotary_tensors(config(),8)
    result=cached.sparse_forward(tokens,cos,sin,anchors)
    x=dense.model.embed_tokens(tokens)
    for layer in dense.model.layers:x=layer(x,cos,sin)
    raw=dense.model.norm(x);z=raw[:,anchors]
    positions=torch.cat([torch.arange(8),anchors]);steps=torch.cat([torch.zeros(8),torch.ones(3)])
    keep=(positions[None,:]<=positions[:,None])&(steps[None,:]<=steps[:,None])
    c=cos.index_select(-2,positions);s=sin.index_select(-2,positions)
    x=torch.cat([dense.model.embed_tokens(tokens),z],1)
    for layer in dense.model.layers:
        attention=layer.self_attn;b,n,_=x.shape;d=attention.dim;heads=attention.heads
        q=attention.q_norm(F.linear(x,attention.q_proj.weight)).view(b,n,heads,d).transpose(1,2)
        k=attention.k_norm(F.linear(x,attention.k_proj.weight)).view(b,n,heads,d).transpose(1,2)
        v=F.linear(x,attention.v_proj.weight).view(b,n,heads,d).transpose(1,2)
        def rotate(t):
            a,bb=t.chunk(2,-1);return t*c+torch.cat([-bb,a],-1)*s
        q,k=rotate(q),rotate(k)
        probability=((q@k.transpose(-2,-1))*(d**-.5)).masked_fill(~keep,float('-inf')).softmax(-1)
        attended=(probability@v).transpose(1,2).reshape(b,n,-1)
        x=x+layer.post_attention_layernorm(F.linear(attended,attention.o_proj.weight))
        gate=F.silu(F.linear(x,layer.mlp.gate_proj.weight));up=F.linear(x,layer.mlp.up_proj.weight)
        x=x+layer.post_feedforward_layernorm(F.linear(gate*up,layer.mlp.down_proj.weight))
    refined=dense.model.norm(x)[:,8:];expected=dense.lm_head(.5*z+.5*refined);raw_logits=dense.lm_head(raw)
    torch.testing.assert_close(result['mixed_logits'],expected,rtol=2e-5,atol=2e-6)
    (result['mixed_logits'].square().mean()+result['raw_logits'].square().mean()).backward()
    (expected.square().mean()+raw_logits.square().mean()).backward()
    for (name,p),(other,q) in zip(cached.named_parameters(),dense.named_parameters(),strict=True):
        assert name==other;torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=3e-6)


def test_static_aot_stages_preserve_cached_outputs_and_all_gradients():
    torch.manual_seed(8851);reference=SparseWorkspaceCandidate(config(),activation_checkpointing=True);compiled=copy.deepcopy(reference)
    compiled.compile_stages('aot_eager')
    tokens=torch.arange(8).reshape(1,-1);cos,sin=rotary_tensors(config(),8);a=torch.tensor([2,5,7]);outputs=[]
    for model in [reference,compiled]:
        r=model.sparse_forward(tokens,cos,sin,a);(r['raw_logits'].square().mean()+r['mixed_logits'].square().mean()).backward();outputs.append(r)
    torch.testing.assert_close(outputs[0]['mixed_logits'],outputs[1]['mixed_logits'],rtol=1e-5,atol=1e-6)
    for p,q in zip(reference.parameters(),compiled.parameters(),strict=True):torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6)
