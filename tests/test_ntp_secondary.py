import copy
import pytest
from compute_trainium.ntp_secondary import ntp_point_contrasts,discovery_decision


def rows(seed=5231):
    # Numerical analysis metadata only;never research examples or model input.
    return [{'cell':cell,'depth':depth,'seed':seed,'training_real_targets':100,
             'training_parameters':10,'deployed_parameters':8,'checkpoint_sha256':cell,
             'metrics':{'weighted_nll':loss}} for cell,depth,loss in [
        ('one-ntp',1,.50),('one-future',1,.49),('recurrent-ntp',1,.48),('recurrent-ntp',2,.45)]]


def test_secondary_needs_both_strong_one_pass_controls_and_same_weight_supported_depth():
    a=rows();points=ntp_point_contrasts(a)
    assert points==pytest.approx({'over_one_ntp':.05,'over_one_future':.04,'supported_depth1_to2':.03})
    assert discovery_decision({s:ntp_point_contrasts(rows(s)) for s in [5231,7283]})['all_point_gates_pass']
    bad=copy.deepcopy(a);bad[1]['metrics']['weighted_nll']=.42
    assert not discovery_decision({5231:ntp_point_contrasts(bad),7283:points})['all_point_gates_pass']
    bad=copy.deepcopy(a);bad[2]['checkpoint_sha256']='other-model'
    with pytest.raises(ValueError,match='identical weights'):ntp_point_contrasts(bad)
    bad=copy.deepcopy(a);bad[0]['training_real_targets']=99
    with pytest.raises(ValueError,match='exposure'):ntp_point_contrasts(bad)


def test_missing_comparisons_cannot_turn_into_a_vacuous_positive():
    with pytest.raises(ValueError,match='contrast'):
        discovery_decision({5231:{},7283:{}})
