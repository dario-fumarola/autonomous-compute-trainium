"""Numerical fixture only:removing a closed training head preserves prediction."""
import torch
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate


def test_lean_normal_decoder_and_full_body_gradients_match_closed_auxiliary_wrapper():
    torch.manual_seed(971);cfg=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,prelude=2,core=2,coda=2,max_seq=32)
    full=FutureSpanMathSDPACandidate(cfg,FutureSpanConfig(context=16,horizon=4,stride=4,weight=0),fp16_body=False)
    lean=ParcaePrimitiveSDPACandidate(cfg,recurrent=True,initialize=False,fp16_body=False);lean.load_state_dict(full.prediction_state_dict(),strict=True)
    ids=torch.randint(41,(1,16));state=torch.randn(1,16,32)*.01
    for n in [1,2,4]:
        full.zero_grad(set_to_none=True);lean.zero_grad(set_to_none=True)
        a,_=full(ids,state,total_steps=n,backward_cap=n);b=lean(ids,state,total_steps=n,backward_cap=n)
        torch.testing.assert_close(a,b,rtol=0,atol=0);a.square().mean().backward();b.square().mean().backward()
        reference={k:p.grad for k,p in full.named_parameters() if not k.startswith('future_lm_head.')}
        for name,p in lean.named_parameters():torch.testing.assert_close(p.grad,reference[name],rtol=0,atol=0)
