import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
from compute_trainium.parcae_candidate import ParcaeCandidateConfig


def test_future_head_and_body_rng_parameters_and_gradients_survive_attention_precision_wrapper():
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    span=FutureSpanConfig(horizon=4,stride=4,context=16)
    torch.manual_seed(7);left=FutureSpanCandidate(config,span,fp16_body=False);rng=torch.get_rng_state()
    torch.manual_seed(7);right=FutureSpanMathSDPACandidate(config,span,fp16_body=False)
    assert torch.equal(rng,torch.get_rng_state())
    assert left.state_dict().keys()==right.state_dict().keys()
    assert all(torch.equal(value,right.state_dict()[key]) for key,value in left.state_dict().items())
    tokens=torch.arange(16)[None];state=torch.randn(1,16,32)*.02
    with sdpa_kernel(SDPBackend.MATH):
        a=left(tokens,state,total_steps=2,backward_cap=2)
        b=right(tokens,state,total_steps=2,backward_cap=2)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        sum(x.square().mean() for x in a).backward()
        sum(x.square().mean() for x in b).backward()
    for (name,x),(other,y) in zip(left.named_parameters(),right.named_parameters()):
        assert name==other
        if x.grad is None:assert y.grad is None
        else:torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
