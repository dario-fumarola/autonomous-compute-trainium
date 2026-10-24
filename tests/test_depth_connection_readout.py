import copy

import pytest

from scripts.analyze_depth_connection import checked_metric


def fixture():
    return {'copy_checksum_and_full_state_readback': True, 'cpu_minus_native_nll': .000001,
        'metrics': {'weights': {'web':.6,'math':.2,'code':.2}, 'weighted_nll': 2.7,
            'domains': {'web': {'nll_sum':60.,'real_targets':20,'nll':3.},
                        'math': {'nll_sum':25.,'real_targets':10,'nll':2.5},
                        'code': {'nll_sum':10.,'real_targets':5,'nll':2.}}}}


def test_domain_token_sums_and_declared_weights():
    assert checked_metric(fixture()) == pytest.approx(2.7)
    wrong = copy.deepcopy(fixture())
    wrong['metrics']['domains']['math']['real_targets'] = 20
    with pytest.raises(ValueError):
        checked_metric(wrong)


def test_failed_recovery_cannot_enter_readout():
    wrong = fixture()
    wrong['copy_checksum_and_full_state_readback'] = False
    with pytest.raises(ValueError):
        checked_metric(wrong)


def test_nonfinite_score_cannot_pass_comparison():
    wrong = fixture()
    wrong['metrics']['weighted_nll'] = float('nan')
    with pytest.raises(ValueError):
        checked_metric(wrong)
