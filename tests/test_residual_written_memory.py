import torch
import pytest
from compute_trainium.residual_written_memory import WrittenMemoryWorkspace,balanced_depth_schedule
from compute_trainium.residual_workspace import ResidualWorkspace


def pair():
    torch.manual_seed(82);a=WrittenMemoryWorkspace(16,8,writes=False).double();b=WrittenMemoryWorkspace(16,8,writes=True).double()
    with torch.no_grad():a.output.weight.normal_(0,.1)
    b.load_state_dict(a.state_dict());return a,b


def test_supported_first_exit_and_earlier_write_intervention():
    a,b=pair();x=torch.randn(2,9,16,dtype=torch.float64);q=torch.tensor([1,4,8]);old=ResidualWorkspace(16,8,mode='one').double();old.load_state_dict(a.state_dict())
    torch.testing.assert_close(a(x,q,depth=1),old(x,q),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(a(x,q,depth=1),b(x,q,depth=1),rtol=0,atol=0)
    torch.testing.assert_close(a(x,q,depth=2),b(x,q,depth=2,suppress_earlier_writes=True),rtol=1e-12,atol=1e-12)
    assert not torch.allclose(b(x,q,depth=2),a(x,q,depth=2))
    with pytest.raises(ValueError):b(x,q,depth=3)


@pytest.mark.parametrize('writes',[False,True])
@pytest.mark.parametrize('depth',[1,2])
def test_reference_all_parameter_gradients_and_causality(writes,depth):
    a,b=pair();a.writes=b.writes=writes;x=torch.randn(2,9,16,dtype=torch.float64,requires_grad=True);q=torch.tensor([0,3,6]);out=a(x,q,depth=depth);ref=b(x.detach(),q,depth=depth,reference=True);torch.testing.assert_close(out,ref,rtol=1e-11,atol=1e-11);out.square().sum().backward();ref.square().sum().backward()
    for (name,p),(_,r) in zip(a.named_parameters(),b.named_parameters(),strict=True):torch.testing.assert_close(p.grad,r.grad,rtol=1e-9,atol=1e-10,msg=name)
    assert torch.count_nonzero(x.grad[:,7:])==0
    changed=x.detach().clone();changed[:,7:]+=100
    torch.testing.assert_close(a(changed,q,depth=depth),out,rtol=0,atol=0)
    torch.testing.assert_close(a(x.detach(),q[1:2],depth=depth),out[:,1:2],rtol=1e-12,atol=1e-12)


def test_zero_output_identity_and_depth_schedule():
    a=WrittenMemoryWorkspace(16,8,writes=True);x=torch.randn(2,9,16);q=torch.tensor([2,7])
    for depth in [1,2]:assert torch.equal(a(x,q,depth=depth),x[:,q])
    schedule=balanced_depth_schedule();assert schedule==balanced_depth_schedule()
    for i in range(0,2048,256):assert schedule[i:i+256].count(1)==128
