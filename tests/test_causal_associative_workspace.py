import torch
import pytest
from compute_trainium.causal_associative_workspace import AssociativeWorkspace,target_mass,mixture_log_probability,sequential_reference


@pytest.mark.parametrize('mode',['bag','cosine','learned'])
def test_causality_full_distribution_and_identity(mode):
    torch.manual_seed(74);m=AssociativeWorkspace(12,4,mode=mode).double();h=torch.randn(2,7,12,dtype=torch.double);labels=torch.randint(0,5,(2,7));anchors=torch.tensor([0,2,5]);targets=torch.randint(0,5,(2,3));raw=torch.full((2,3),-2.,dtype=torch.double)
    torch.testing.assert_close(m(h,labels,anchors,targets,raw),raw,rtol=0,atol=0)
    with torch.no_grad():m.strength.fill_(.4)
    for a in anchors.tolist():
        q=torch.tensor([a]);weights,gate=m.read(h,q);full=torch.zeros(2,1,5,dtype=torch.double);full.scatter_add_(-1,labels[:,None,:],weights)
        mass=target_mass(weights,labels,q,targets[:,:1]);torch.testing.assert_close(mass,full.gather(-1,targets[:,:1,None]).squeeze(-1))
        changed=labels.clone();changed[:,a:]=(changed[:,a:]+1)%5
        torch.testing.assert_close(m(h,changed,q,targets[:,:1],raw[:,:1]),m(h,labels,q,targets[:,:1],raw[:,:1]),rtol=0,atol=0)
        other=h.clone();other[:,a+1:]+=100
        torch.testing.assert_close(m(other,labels,q,targets[:,:1],raw[:,:1]),m(h,labels,q,targets[:,:1],raw[:,:1]),rtol=0,atol=0)
        if a==0:assert torch.count_nonzero(weights)==0 and torch.count_nonzero(gate)==0


def test_rotation_preserves_causal_bag_and_gradients():
    torch.manual_seed(75);m=AssociativeWorkspace(12,4,mode='bag').double();m.strength.data.fill_(.4);h=torch.randn(2,7,12,dtype=torch.double);labels=torch.randint(0,5,(2,7));q=torch.tensor([1,3,6]);targets=torch.randint(0,5,(2,3));raw=torch.full((2,3),-2.,dtype=torch.double)
    torch.testing.assert_close(m(h,labels,q,targets,raw),m(h,labels,q,targets,raw,rotate=True),rtol=1e-14,atol=1e-14)
    for mode in ['bag','cosine','learned']:
        m=AssociativeWorkspace(12,4,mode=mode).double();m.strength.data.fill_(.4);(-m(h,labels,q,targets,raw).mean()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_projected_zero_boundary_can_learn_and_mixture_is_normalized():
    raw=torch.tensor([[.1,.2,.7]],dtype=torch.double);cache=torch.tensor([[.7,.1,.2]],dtype=torch.double);gate=torch.tensor(.2,dtype=torch.double,requires_grad=True);p=mixture_log_probability(raw.log(),cache,gate).exp();torch.testing.assert_close(p.sum(),torch.tensor(1.,dtype=torch.double));(-p[0,0].log()).backward();assert gate.grad<0
    m=AssociativeWorkspace(4,2,mode='bag');m.strength.data.fill_(-1);m.project_();assert m.strength==0
    (-m.strength.clamp(0,1)).backward();assert m.strength.grad==-1
    logs=torch.tensor([-1e-20,-.00112345,-8.123456],dtype=torch.float32).double();gate=torch.zeros(3,dtype=torch.double,requires_grad=True);mass=torch.tensor([.2,.3,.4],dtype=torch.double);result=mixture_log_probability(logs,mass,gate);assert torch.equal(result,logs);result.sum().backward();torch.testing.assert_close(gate.grad,mass/logs.exp()-1,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('mode',['bag','cosine','learned'])
def test_independent_prefix_reference_outputs_and_gradients(mode):
    torch.manual_seed(76);a=AssociativeWorkspace(12,4,mode=mode).double();a.strength.data.fill_(.4);b=AssociativeWorkspace(12,4,mode=mode).double();b.load_state_dict(a.state_dict());h=torch.randn(2,7,12,dtype=torch.double);labels=torch.randint(0,5,(2,7));q=torch.tensor([0,2,6]);targets=labels[:,q];raw=torch.full((2,3),-2.,dtype=torch.double)
    x=a(h,labels,q,targets,raw);y=sequential_reference(b,h,labels,q,targets,raw);torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12);x.sum().backward();y.sum().backward()
    for (name,p),(_,r) in zip(a.named_parameters(),b.named_parameters(),strict=True):torch.testing.assert_close(p.grad,r.grad,rtol=1e-9,atol=1e-10,msg=name)
