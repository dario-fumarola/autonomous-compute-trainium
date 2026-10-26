import pytest
from scripts.qualify_baseline_managed import validate_phase
from scripts.run_baseline_numerical import recovery_gate


def plan(profile='mixing'):
    return dict(purpose='baseline_native_qualification_only',root_comm_ports={'owned':17000},
                phases={'owned':dict(profile=profile,mode='warm')})


@pytest.mark.parametrize('profile,arm',[('one','fixed1'),('mixing','poisson2'),('ordinary8','ordinary')])
def test_explicit_baseline_phase(profile,arm):validate_phase(plan(profile),'owned',17000,profile=profile,arm=arm,mode='warm')


@pytest.mark.parametrize('profile,arm,mode,port',[('mixing','fixed1','warm',17000),
    ('ordinary8','ordinary','replay',17000),('one','fixed1','warm',17001)])
def test_reject_changed_profile_depth_or_phase(profile,arm,mode,port):
    with pytest.raises(ValueError):validate_phase(plan(profile),'owned',port,profile=profile,arm=arm,mode=mode)


def recovered(gain=.009):
    return dict(status='COMPLETE_ORIGINAL_ONE_CONTROL_RECOVERY',
        scores={'one-s17293-data':{},'one-s17293-work':{}},
        gradient_gates={key:dict(independent_cpu=dict(status='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'))
                        for key in ['stable-prefix','one-s17293-data','one-s17293-work']},
        comparisons={'one-s17293-work':dict(gain_nats=gain)})


def test_baseline_remediation_gate_requires_both_positive_and_useful_mean():
    assert recovery_gate(recovered(),.012)
    assert not recovery_gate(recovered(.005),.012)
    assert not recovery_gate(recovered(-.001),.023)


def test_baseline_gate_cannot_advance_partial_or_numerically_failed_recovery():
    value=recovered();value['gradient_gates'].pop('stable-prefix')
    assert not recovery_gate(value,.012)
    value=recovered();value['gradient_gates']['one-s17293-work']['independent_cpu']['status']='FAIL'
    assert not recovery_gate(value,.012)
