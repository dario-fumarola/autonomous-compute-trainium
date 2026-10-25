import torch
from compute_trainium.natural_chain_native_loss import answer_parts
from compute_trainium.natural_chain_objective import answer_parts as reference

def test_native_explicit_cross_entropy_matches_reference_values_gradients_and_fullgraph_aot():
    torch.manual_seed(197);logits=torch.randn(2,16,23,requires_grad=True);target=torch.randint(23,(2,16));mask=torch.rand(2,16)>.7;stop=torch.zeros_like(mask);stop[:,-1]=True;mask[:,-1]=False
    expected=reference(logits,target,mask,stop);actual=answer_parts(logits,target,mask,stop)
    torch.testing.assert_close(expected,actual,rtol=1e-6,atol=2e-6)
    left=torch.autograd.grad(expected.sum(),logits)[0];right=torch.autograd.grad(actual.sum(),logits)[0];torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-7)
    compiled=torch.compile(answer_parts,backend='aot_eager',fullgraph=True,dynamic=False)
    torch.testing.assert_close(compiled(logits,target,mask,stop),actual)
