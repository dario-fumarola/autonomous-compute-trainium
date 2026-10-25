import torch

from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig
from compute_trainium.future_span_reference_optimizer import reference_optimizer,fp32_reference_execution
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_recovery_check import moments
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer,changing_qualification_schedule
from compute_trainium.parcae_update_qualification import named_moments


def test_new_head_and_all_historical_moments_restore_by_name_for_independent_next_step():
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    span=FutureSpanConfig(context=16,horizon=4,stride=3)
    torch.manual_seed(271)
    primary=FutureSpanCandidate(config,span,fp16_body=False)
    optimizer=ParcaeRuntimeScalarOptimizer(primary,gate_policy='release_muon',fused_adamw=False)
    for update in range(3):
        changing_qualification_schedule(optimizer,update)
        for p in primary.parameters():p.grad=torch.randn_like(p)*.01
        optimizer.step()
    other=FutureSpanCandidate(config,span,initialize=False,fp16_body=False)
    other.load_state_dict(primary.state_dict())
    saved=optimizer.state_dict()
    reference=reference_optimizer(other,saved['recipe']['groups'],saved)
    restored=named_moments(other,reference,reference=True)
    original=moments(saved)
    for name in original:
        for key in original[name]:
            if isinstance(original[name][key],torch.Tensor):
                torch.testing.assert_close(restored[name][key],original[name][key],rtol=0,atol=0)
            else:assert restored[name][key]==original[name][key]
    changing_qualification_schedule(optimizer,3);changing_qualification_schedule(reference,3)
    for name,p in primary.named_parameters():
        p.grad=torch.randn_like(p)*.01
        dict(other.named_parameters())[name].grad=p.grad.clone()
    optimizer.step()
    with fp32_reference_execution():reference.step()
    for name,p in primary.named_parameters():
        torch.testing.assert_close(p,dict(other.named_parameters())[name],atol=1e-7,rtol=1e-6)
    assert 'future_lm_head.weight' in original and 'future_lm_head.bias' in original
