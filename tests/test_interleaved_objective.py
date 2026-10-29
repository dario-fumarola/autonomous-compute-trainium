import torch
import pytest
from compute_trainium.interleaved_objective import target_loss_sum,reference_loss_sum
from test_interleaved_workspace import setup

@pytest.mark.parametrize('mode',['ordinary','pause','latent'])
def test_masked_stable_objective_matches_independent_reference(mode):
    m,x,y=setup(True);n,_,_=setup(False);n.load_state_dict(m.state_dict())
    mask=torch.ones_like(y);mask[0,2:]=0
    a=target_loss_sum(m,x,y,mask,2,mode);b=reference_loss_sum(n,x,y,mask,2,mode)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10);a.backward();b.backward()
    for p,q in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)
