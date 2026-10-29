"""Numerical fixtures only; no generated research examples."""
import copy
import pytest
import torch

from compute_trainium.chunked_persistent_workspace import ChunkedPersistentWorkspace,matrix_work
from compute_trainium.model import ModelConfig
from compute_trainium.tied_interleaved_ordinary import TiedOrdinary

TOKENS=torch.tensor([[2,5,8,11,14],[3,6,9,12,15]])


def make(dtype=torch.float64,explicit=False):
    torch.manual_seed(621)
    c=ModelConfig(dim=16,layers=2,heads=4,kv_heads=2,ffn_dim=24,vocab_size=31,max_seq=8)
    return ChunkedPersistentWorkspace(c,explicit_attention=explicit).to(dtype)


def loss(logits):
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),((TOKENS+1)%31).flatten())


def gradients(a,b,atol,rtol):
    assert a.state_dict().keys()==b.state_dict().keys()
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,atol=atol,rtol=rtol,msg=name)


@pytest.mark.parametrize('steps',[0,1,2])
@pytest.mark.parametrize('mode',['latent','pause'])
@pytest.mark.parametrize('dtype,explicit',[(torch.float64,False),(torch.float64,True),(torch.float32,True)])
def test_cached_reference_states_logits_and_gradients(steps,mode,dtype,explicit):
    model=make(dtype,explicit);reference=copy.deepcopy(model)
    actual,states=model.cached(TOKENS,2,steps,mode,return_states=True)
    expected,other=reference.full_reference(TOKENS,2,steps,mode,return_states=True)
    atol,rtol=(1e-11,1e-9) if dtype==torch.float64 else (4e-6,2e-4)
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    assert len(states)==len(other)==2*steps
    for x,y in zip(states,other):torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    loss(actual).backward();loss(expected).backward();gradients(model,reference,atol,rtol)


def test_disabled_workspace_is_exact_ordinary_through_optimizer():
    model=make();torch.manual_seed(621);ordinary=TiedOrdinary(model.config).double()
    assert sum(p.numel() for p in model.parameters())==model.config.parameter_count()
    a,b=model(TOKENS,2,0),ordinary(TOKENS)
    assert torch.equal(a,b)
    loss(a).backward();loss(b).backward();gradients(model,ordinary,0,0)
    x=torch.optim.AdamW(model.parameters(),lr=.001);y=torch.optim.AdamW(ordinary.parameters(),lr=.001)
    x.step();y.step()
    for name,p in model.state_dict().items():assert torch.equal(p,ordinary.state_dict()[name])
    for u,v in zip(x.state.values(),y.state.values()):
        for key in u:assert torch.equal(u[key],v[key])


@pytest.mark.parametrize('steps',[1,2])
def test_prefix_consistency_including_incomplete_chunk(steps):
    model=make();full=model(TOKENS,2,steps)
    for length in range(1,6):
        short=model(TOKENS[:,:length],2,steps)
        torch.testing.assert_close(full[:,:length],short,atol=1e-12,rtol=1e-10)
    altered=TOKENS.clone();altered[:,3:]=30
    torch.testing.assert_close(model(altered,2,steps)[:,:3],full[:,:3],atol=0,rtol=0)


def test_detached_history_is_wrong_even_with_identical_forward():
    model=make();broken=copy.deepcopy(model)
    a=model.cached(TOKENS,2,1);b=broken.cached(TOKENS,2,1,detach_cache=True)
    assert torch.equal(a,b)
    loss(a).backward();loss(b).backward()
    assert sum((p.grad-q.grad).square().sum() for p,q in zip(model.parameters(),broken.parameters())).sqrt()>1e-4


def test_generated_state_reaches_later_real_tokens():
    model=make()
    generated=model(TOKENS,2,1,'latent');constant=model(TOKENS,2,1,'pause')
    torch.testing.assert_close(generated[:,:1],constant[:,:1],atol=0,rtol=0)
    # Position two is an ordinary source readout after the first workspace,
    # not the prediction made directly from a workspace slot.
    assert (generated[:,2]-constant[:,2]).abs().max()>1e-6


@pytest.mark.parametrize('steps',[0,1,2])
def test_cache_rectangles_and_real_target_head(steps):
    c=make().config
    sizes=[2]+[1]*steps+[2]+[1]*steps+[1]
    stored=0;attention=0
    for size in sizes:
        attention+=2*4*c.layers*c.dim*size*(stored+size);stored+=size
    got=matrix_work(c,5,2,steps,2)
    assert got['attention_forward_flops']==attention
    assert got['stored_positions']==5+2*steps
    assert got['sequential_body_calls']==len(sizes)
    assert got['head_forward_flops']==2*2*5*c.dim*c.vocab_size
    full=matrix_work(c,5,2,0,2,ordinary_full=True)
    assert full['forward_matmul_flops']==c.flops(5,5,2)['forward_matmul_flops']
    if steps==0:assert got['forward_matmul_flops']<full['forward_matmul_flops']


@pytest.mark.parametrize('stride,steps',[(0,1),(True,1),(2,-1),(2,3),(2,True)])
def test_bad_budget_fails_closed(stride,steps):
    with pytest.raises(ValueError):make()(TOKENS,stride,steps)


@pytest.mark.parametrize('steps,mode',[(0,'latent'),(1,'latent'),(1,'pause'),(2,'latent')])
def test_static_frontend_captures_full_gradient(steps,mode):
    model=make(torch.float32,True);reference=copy.deepcopy(model)
    compiled=torch.compile(model,backend='eager',fullgraph=True)
    a,b=compiled(TOKENS,2,steps,mode),reference(TOKENS,2,steps,mode)
    torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    loss(a).backward();loss(b).backward();gradients(model,reference,1e-7,1e-6)
