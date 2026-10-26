import copy
import pytest
from compute_trainium.lean_ntp_admission import passes_development_gate


def study():
    return dict(status='COMPLETE_FIXED_CONTROL_DEVELOPMENT_REPLICATION',test_accessed=False,
                terminal_alpha_spent=0,terminal_success=False,
                comparisons={f'{s}-{k}':{'gain_nats':.012} for s in [5231,17293]
                             for k in ['equal_data','one_work','cheap_work']})


def test_no_qualification_from_incomplete_or_selected_comparisons():
    r=study();assert passes_development_gate(r)
    r['status']='RUNNING'
    with pytest.raises(ValueError,match='complete'):passes_development_gate(r)
    r=study();del r['comparisons']['17293-cheap_work']
    with pytest.raises(ValueError,match='retained'):passes_development_gate(r)


def test_both_work_controls_and_seeds_must_pass_without_rounding_or_claimed_means():
    r=study();r['comparisons']['17293-cheap_work']['gain_nats']=-.001
    r['descriptive_two_seed_mean_gains']={'one_work':1.,'cheap_work':1.}
    assert not passes_development_gate(r)
    r=study();r['comparisons']['17293-one_work']['gain_nats']=.0079
    assert not passes_development_gate(r)
    r=study();r['comparisons']['17293-one_work']['gain_nats']=.008
    assert passes_development_gate(r)


def test_no_confirmation_leak_or_nonfinite_metric():
    for key,value in [('test_accessed',True),('terminal_alpha_spent',.025),('terminal_success',True)]:
        r=study();r[key]=value
        with pytest.raises(ValueError,match='development'):passes_development_gate(r)
    r=study();r['comparisons']['5231-one_work']['gain_nats']=float('nan')
    with pytest.raises(ValueError,match='Finite'):passes_development_gate(r)
