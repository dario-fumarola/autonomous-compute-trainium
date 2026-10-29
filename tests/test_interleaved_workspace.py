import copy
import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace,matrix_work

def setup(explicit=False):
    torch.manual_seed(9821)
    c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=8)
    m=InterleavedWorkspace(c,explicit_attention=explicit).double()
    x=torch.randint(0,23,(2,5));y=torch.randint(0,23,(2,5))
    return m,x,y

def test_sequential_convergence_forward_and_full_gradient():
    m,x,y=setup();a,z=m.jacobi(x,4,return_states=True);b,zs=m.sequential(x,return_states=True)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(z,zs,atol=1e-10,rtol=1e-10)
    F.cross_entropy(a.flatten(0,1),y.flatten()).backward()
    gradients={n:p.grad.clone() for n,p in m.named_parameters()};m.zero_grad()
    F.cross_entropy(b.flatten(0,1),y.flatten()).backward()
    for n,p in m.named_parameters():torch.testing.assert_close(gradients[n],p.grad,atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize('k',[0,1,2,3])
def test_partial_stabilization_and_future_invariance(k):
    m,x,y=setup();a=m.jacobi(x,k);b=m.sequential(x)
    torch.testing.assert_close(a[:,:k+1],b[:,:k+1],atol=1e-10,rtol=1e-10)
    changed=x.clone();changed[:,3:]=(changed[:,3:]+1)%23
    torch.testing.assert_close(a[:,:3],m.jacobi(changed,k)[:,:3],atol=0,rtol=0)

def test_ordinary_identity_mask_and_independent_attention_gradients():
    m,x,y=setup(True);n,_,_=setup(False);n.load_state_dict(m.state_dict())
    ordinary=F.linear(m.body_at(m.embeddings(x),torch.arange(x.shape[1])),m.prediction_weight)
    torch.testing.assert_close(ordinary,m(x,mode='ordinary'),atol=0,rtol=0)
    mask=torch.ones_like(y);mask[0,-2:]=0
    for model in (m,n):
        logits=model(x,2);loss=(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').view_as(y)*mask).sum()/mask.sum();loss.backward()
    for p,q in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)
    changed=y.clone();changed[mask==0]=(changed[mask==0]+1)%23
    def loss(target):return (F.cross_entropy(m(x).flatten(0,1),target.flatten(),reduction='none').view_as(target)*mask).sum()
    assert torch.equal(loss(y),loss(changed))

def test_generated_state_gradient_and_previous_state_use():
    m,x,y=setup();source=m.embeddings(x);pos=torch.arange(x.shape[1]);z=m.body_at(source,pos)
    z.retain_grad();out=m.body_at(m.interleave(source,z),pos.repeat_interleave(2))[:,1::2]
    F.cross_entropy(F.linear(out,m.prediction_weight)[:,-1],y[:,-1]).backward()
    # A later prediction can use earlier generated states, not only its own.
    assert z.grad[:,:-1].norm()>1e-10
    assert m.embedding.weight.grad.norm()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

@pytest.mark.parametrize('mode,k',[('ordinary',0),('pause',0),('latent',0),('latent',2)])
def test_exact_dispatched_matrix_work(mode,k):
    from torch.utils._python_dispatch import TorchDispatchMode
    class Counter(TorchDispatchMode):
        def __init__(self):super().__init__();self.flops=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func is torch.ops.aten.mm.default:
                a,b=args;self.flops+=2*a.shape[0]*a.shape[1]*b.shape[1]
            elif func is torch.ops.aten.bmm.default:
                a,b=args;self.flops+=2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
            return func(*args,**(kwargs or {}))
    m,x,y=setup(True);counter=Counter()
    with counter:
        loss=F.cross_entropy(m(x,k,mode).flatten(0,1),y.flatten());forward=counter.flops;loss.backward()
    expected=matrix_work(m.config,5,k,mode,batch=2)
    assert (forward,counter.flops)==(expected['forward_matmul_flops'],expected['training_matmul_flops'])

def test_exact_optimizer_replay():
    m,x,y=setup();o=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False)
    def step(model,opt):
        opt.zero_grad();F.cross_entropy(model(x).flatten(0,1),y.flatten()).backward();opt.step()
    step(m,o);n=copy.deepcopy(m);p=torch.optim.AdamW(n.parameters(),lr=.001,foreach=False);p.load_state_dict(copy.deepcopy(o.state_dict()))
    step(m,o);step(n,p)
    for a,b in zip(m.parameters(),n.parameters()):assert torch.equal(a,b)
    for a,b in zip(o.state.values(),p.state.values()):
        for key in a:assert torch.equal(a[key],b[key])

@pytest.mark.parametrize('k,mode',[(True,'latent'),(-1,'latent'),(2,'unknown')])
def test_invalid_specs(k,mode):
    m,x,y=setup()
    with pytest.raises(ValueError):m(x,k,mode)
    with pytest.raises(ValueError):matrix_work(m.config,5,k,mode)
