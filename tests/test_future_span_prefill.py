import copy
import pytest
import torch

from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_evaluation import prediction_state
from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
from compute_trainium.future_span_prefill import readout, check_readout, prefill_budget, validate_phase
from compute_trainium.parcae_candidate import ParcaeCandidateConfig, matmul_ledger
from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate


@pytest.mark.parametrize('fp16',[False,True])
def test_head_removed_inference_equals_trained_path_at_all_supported_depths(fp16):
    # Numerical function/immutability fixtures,not generated research examples.
    torch.manual_seed(17)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,prelude=1,core=2,coda=1,max_seq=16)
    full=FutureSpanMathSDPACandidate(config,FutureSpanConfig(context=16,horizon=4,stride=2),
        initialize=False,fp16_body=fp16).eval()
    bare=ParcaePrimitiveSDPACandidate(config,initialize=False,fp16_body=fp16).eval().requires_grad_(False)
    bare.load_state_dict(prediction_state(full.state_dict()),strict=True)
    before=copy.deepcopy(bare.state_dict())
    tokens=torch.arange(16)[None]%41;noise=torch.randn(1,16,32)
    old_noise=noise.clone()
    with torch.inference_mode():
        for depth in [1,2,3]:
            expected,_=full(tokens,noise,total_steps=depth,backward_cap=depth)
            actual=bare(tokens,noise,total_steps=depth,backward_cap=depth)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            total,samples=readout(actual,tokens,torch.ones_like(tokens,dtype=torch.bool))
            assert check_readout(total,samples,total,samples,tokens.numel())['passes']
            with pytest.raises(ValueError):check_readout(total+1,samples,total,samples,tokens.numel())
    torch.testing.assert_close(noise,old_noise,rtol=0,atol=0)
    assert not any(p.grad is not None or p.requires_grad for p in bare.parameters())
    for key,value in bare.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_prefill_budget_requires_real_warm_measurements_and_complete_rounds():
    hardware={'status':'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION','records':[
        {'case':{'identity':c},'native':{'warm':{'warm_updates':100,'warm_seconds':160.,'wall_seconds':325.}}}
        for c in ['one-ntp','one-future','recurrent-ntp','recurrent-future']]}
    result=prefill_budget(hardware)
    assert result['fits_cap'] and result['two_rounds_plus_cpu_seconds_allowance']<1800
    hardware['records'][0]['native']['warm']['warm_updates']=12
    with pytest.raises(ValueError):prefill_budget(hardware)


def test_phase_guard_rejects_unsupported_depth_missing_work_or_numerical_failure():
    from dataclasses import asdict
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,prelude=1,core=2,coda=1)
    plan={'model':asdict(c),'cases':{'candidate':{'checkpoint_sha256':'weights','depths':[1,2,3]}}}
    row=dict(status='PASS_READ_ONLY_NATIVE_PREFILL_TIMING',case_id='candidate',round=1,
        plan_sha256='plan',checkpoint_sha256='weights',deployed_parameters=c.parameter_count(True),
        head_removed=True,weights_unchanged=True,unique_windows=64,context=1024,world=32,
        optimizer_updates=0,gradients_created=False,test_accessed=False,physical_chips=list(range(8)),
        records=[dict(depth=d,warm_forwards_per_rank=100,timed_forwards_per_rank=100,actual_forwards=204*32,
            forward_matrix_ops_per_call=matmul_ledger(c,1024,total_steps=d)['forward_matmul_flops'],
            timed_forward_median_seconds=.1,timed_forward_p95_seconds=.2,
            maximum_absolute_nll_error=.001,maximum_logit_relative_l2=.001) for d in [1,2,3]])
    validate_phase(row,plan,'candidate',1,'plan')
    for kind in ['depth','count','precision','weights']:
        bad=copy.deepcopy(row)
        if kind=='depth':bad['records'][-1]['depth']=4
        elif kind=='count':bad['records'][0]['actual_forwards']=100
        elif kind=='precision':bad['records'][0]['maximum_absolute_nll_error']=.004
        else:bad['weights_unchanged']=False
        with pytest.raises(ValueError):validate_phase(bad,plan,'candidate',1,'plan')
