import copy
import pytest
import torch
from compute_trainium.model import ModelConfig
from compute_trainium.untied_probability_workspace import UntiedProbabilityWorkspace
from compute_trainium.probability_objective import target_loss_sum,reference_loss_sum,exit_logits
from compute_trainium.probability_workspace import matrix_work
from compute_trainium.stable_target_loss import stable_target_log_probs


def setup(explicit=False):
    torch.manual_seed(77)
    c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=12)
    m=UntiedProbabilityWorkspace(c,explicit_attention=explicit).double()
    x=torch.randint(23,(2,8));y=torch.randint(23,(2,8));mask=torch.ones_like(y);mask[0,-2:]=0
    return m,x,y,mask


@pytest.mark.parametrize('depth',[0,1,2])
def test_default_loss_gradient_and_optimizer_preserve_old_path(depth):
    m,x,y,mask=setup();n=copy.deepcopy(m)
    a=-(stable_target_log_probs(m(x,depth),y)*mask).sum()
    b=target_loss_sum(n,x,y,mask,depth)
    assert torch.equal(a,b);a.backward();b.backward()
    oa=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False);ob=torch.optim.AdamW(n.parameters(),lr=.001,foreach=False)
    for p,q in zip(m.parameters(),n.parameters()):assert torch.equal(p.grad,q.grad)
    oa.step();ob.step()
    for p,q in zip(m.parameters(),n.parameters()):assert torch.equal(p,q)
    for s,t in zip(oa.state.values(),ob.state.values()):
        for k in s:assert torch.equal(s[k],t[k])


@pytest.mark.parametrize('depth',[0,1])
def test_joint_independent_reference_gradients_causality_and_supported_exits(depth):
    m,x,y,mask=setup(True);n,_,_,_=setup(False);n.load_state_dict(m.state_dict())
    a=target_loss_sum(m,x,y,mask,depth,objective='uniform_exits')
    b=reference_loss_sum(n,x,y,mask,depth,objective='uniform_exits')
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11);a.backward();b.backward()
    for p,q in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-10)
    exits=exit_logits(m,x,depth,'expected','uniform_exits')
    for k,logits in enumerate(exits):torch.testing.assert_close(logits,m(x,k),rtol=0,atol=0)
    changed=x.clone();changed[:,5:]=(changed[:,5:]+1)%23
    for a,b in zip(exits,exit_logits(m,changed,depth,'expected','uniform_exits')):
        torch.testing.assert_close(a[:,:5],b[:,:5],rtol=0,atol=0)
    if depth==0:assert torch.equal(a[:,:5],m(x,0)[:,:5])


def test_uniform_position_weights_mask_and_full_proposal_gradient():
    m,x,y,mask=setup();a,b=exit_logits(m,x,1,'expected','uniform_exits')
    a.retain_grad();b.retain_grad()
    loss=-.5*((stable_target_log_probs(a,y)+stable_target_log_probs(b,y))*mask).sum()
    torch.testing.assert_close(loss,target_loss_sum(m,x,y,mask,1,objective='uniform_exits'),rtol=0,atol=1e-14)
    expected=-.5*(stable_target_log_probs(a,y)*mask).sum()
    direct=torch.autograd.grad(expected,a,retain_graph=True)[0]
    loss.backward()
    # Deep loss propagates through proposal probabilities, not only direct CE.
    assert (a.grad-direct).norm()>1e-8
    assert torch.count_nonzero(b.grad[mask==0])==0
    altered=y.clone();altered[mask==0]=(altered[mask==0]+1)%23
    assert torch.equal(target_loss_sum(m,x,y,mask,1,objective='uniform_exits'),target_loss_sum(m,x,altered,mask,1,objective='uniform_exits'))


def test_joint_next_update_exact_restore_and_executed_matrix_budget():
    m,x,y,mask=setup(True)
    def step(q,o):
        o.zero_grad();target_loss_sum(q,x,y,mask,1,objective='uniform_exits').backward();o.step()
    o=torch.optim.AdamW(m.parameters(),lr=.001,foreach=False);step(m,o)
    n=copy.deepcopy(m);p=torch.optim.AdamW(n.parameters(),lr=.001,foreach=False);p.load_state_dict(copy.deepcopy(o.state_dict()))
    step(m,o);step(n,p)
    for a,b in zip(m.parameters(),n.parameters()):assert torch.equal(a,b)
    for a,b in zip(o.state.values(),p.state.values()):
        for k in a:assert torch.equal(a[k],b[k])
    from torch.utils._python_dispatch import TorchDispatchMode
    class Count(TorchDispatchMode):
        def __init__(self):super().__init__();self.flops=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func is torch.ops.aten.mm.default:
                a,b=args;self.flops+=2*a.shape[0]*a.shape[1]*b.shape[1]
            elif func is torch.ops.aten.bmm.default:
                a,b=args;self.flops+=2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
            return func(*args,**(kwargs or {}))
    results=[]
    for objective in ('final_only','uniform_exits'):
        m.zero_grad();c=Count()
        with c:
            value=target_loss_sum(m,x,y,mask,1,objective=objective);forward=c.flops;value.backward()
        results.append((forward,c.flops))
    expected=matrix_work(m.config,8,1,batch=2)
    assert results[0]==results[1]
    assert results[1]==(expected['forward_matmul_flops'],expected['training_matmul_flops'])


@pytest.mark.parametrize('depth,proposal,objective',[(2,'expected','uniform_exits'),(1,'uniform','uniform_exits'),(1,'expected','unknown'),(True,'expected','uniform_exits')])
def test_rejects_unqualified_objective_specs(depth,proposal,objective):
    m,x,y,mask=setup()
    with pytest.raises(ValueError):target_loss_sum(m,x,y,mask,depth,proposal,objective)
    with pytest.raises(ValueError):reference_loss_sum(m,x,y,mask,depth,proposal,objective)
