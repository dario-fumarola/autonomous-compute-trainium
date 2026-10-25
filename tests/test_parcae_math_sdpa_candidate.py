import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_math_sdpa_candidate import ParcaeMathSDPACandidate,math_sdpa


def test_upcast_sdpa_matches_pytorch_math_at_saturated_half_precision_self_attention():
    g=torch.Generator().manual_seed(109)
    base=F.rms_norm(torch.randn(1,2,32,128,generator=g),(128,)).half()
    original=[base.clone().requires_grad_(),base.clone().requires_grad_(),torch.randn(base.shape,generator=g).half().requires_grad_()]
    candidate=[t.detach().clone().requires_grad_() for t in original];gradient=torch.randn(base.shape,generator=g).half()
    with sdpa_kernel(SDPBackend.MATH):
        a=F.scaled_dot_product_attention(*original,is_causal=True);b=math_sdpa(*candidate)
        a.backward(gradient);b.backward(gradient)
    torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-3)
    for x,y in zip(original,candidate):torch.testing.assert_close(x.grad,y.grad,rtol=.002,atol=2e-5)


def test_model_replacement_preserves_parameters_rng_and_cpu_default_math_function():
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    for recurrent in [False,True]:
        torch.manual_seed(7);a=ParcaeFP16Candidate(c,recurrent=recurrent);end_a=torch.get_rng_state()
        torch.manual_seed(7);b=ParcaeMathSDPACandidate(c,recurrent=recurrent);end_b=torch.get_rng_state()
        assert torch.equal(end_a,end_b)
        assert dict(a.state_dict()).keys()==dict(b.state_dict()).keys()
        assert all(torch.equal(t,b.state_dict()[n]) for n,t in a.state_dict().items())
        tokens=torch.arange(16)[None];state=torch.randn(1,16,32)*.02 if recurrent else None
        with sdpa_kernel(SDPBackend.MATH):
            left=a(tokens,state,total_steps=2 if recurrent else 1,backward_cap=2 if recurrent else 1)
            right=b(tokens,state,total_steps=2 if recurrent else 1,backward_cap=2 if recurrent else 1)
        torch.testing.assert_close(left,right,rtol=.002,atol=.002)
