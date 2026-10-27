import torch

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
from compute_trainium.scale_precision_probe import FP32BoundaryCandidate,probe_policy,precision_model


def test_boundary_intervention_has_identical_fp32_function_shared_core_and_gradient():
    torch.manual_seed(781)
    config=ParcaeCandidateConfig(vocab_size=128,dim=64,heads=4,ffn_dim=128,max_seq=16)
    original=ParcaePrimitiveSDPACandidate(config,recurrent=True,fp16_body=False)
    changed=FP32BoundaryCandidate(config,recurrent=True,fp16_body=False)
    changed.load_state_dict(original.state_dict(),strict=True)
    tokens=torch.arange(8).reshape(1,8);state=torch.randn(1,8,64)*.01
    a=original(tokens,state,total_steps=2,backward_cap=2)
    b=changed(tokens,state,total_steps=2,backward_cap=2)
    assert torch.equal(a,b)
    a.square().mean().backward();b.square().mean().backward()
    for (an,ap),(bn,bp) in zip(original.named_parameters(),changed.named_parameters(),strict=True):
        assert an==bn and torch.equal(ap.grad,bp.grad)
    assert changed.lm_head.weight is changed.transformer.wte.weight
    assert probe_policy('scaled16')[2]==65536


def test_boundary_policy_keeps_ordinary_computation_and_gradients_exact():
    for half in [False,True]:
        torch.manual_seed(782)
        config=ParcaeCandidateConfig(vocab_size=128,dim=64,heads=4,ffn_dim=128,max_seq=16)
        original=ParcaePrimitiveSDPACandidate(config,recurrent=False,fp16_body=half)
        changed=FP32BoundaryCandidate(config,recurrent=False,fp16_body=half)
        changed.load_state_dict(original.state_dict(),strict=True)
        tokens=torch.arange(8).reshape(1,8)
        a=original(tokens);b=changed(tokens)
        assert torch.equal(a,b)
        a.square().mean().backward();b.square().mean().backward()
        for (an,ap),(bn,bp) in zip(original.named_parameters(),changed.named_parameters(),strict=True):
            assert an==bn and torch.equal(ap.grad,bp.grad)
        assert changed.lm_head.weight is changed.transformer.wte.weight
    assert precision_model('fp16_body') is ParcaePrimitiveSDPACandidate
    assert precision_model('fp32_recurrent_boundaries') is FP32BoundaryCandidate
