"""Numerical statistical fixtures, not generated research examples."""
import copy
import math

import pytest

from compute_trainium.terminal_confirmation import allocated_alpha, fixed_sample_result, validate_registration


def registration(attempt=1):
    return {'attempt': attempt, 'alpha': allocated_alpha(attempt),
            'primary_comparisons': 1, 'paired_seeds': [1001, 1002, 1003, 1004, 1005],
            'minimum_gain_nats': .01,
            **{key + '_sha256': str(index) * 64 for index, key in enumerate(
                ['candidate', 'control', 'protocol', 'analysis', 'panel'], start=1)}}


def test_alpha_sequence_cannot_reset_and_sums_to_family_budget():
    assert allocated_alpha(1) == .025
    assert sum(allocated_alpha(k) for k in range(1, 101)) == pytest.approx(.05)
    for invalid in [0, -1, True, 1.5, 1001]:
        with pytest.raises(ValueError): allocated_alpha(invalid)
    with pytest.raises(ValueError):
        validate_registration(registration(2), [], used_seeds=[], used_panel_shas=[])


def test_registration_requires_new_seeds_panel_and_immutable_primary_freezes():
    spec = registration()
    assert validate_registration(spec, [], used_seeds=[11, 23], used_panel_shas=['a' * 64])
    for changes in [{'paired_seeds': spec['paired_seeds'][:4]},
                    {'paired_seeds': [1001] * 5}, {'protocol_sha256': 'unfrozen'},
                    {'primary_comparisons': 2}, {'minimum_gain_nats': .001}, {'alpha': .05}]:
        with pytest.raises(ValueError):
            validate_registration(dict(spec, **changes), [], used_seeds=[], used_panel_shas=[])
    with pytest.raises(ValueError):
        validate_registration(spec, [], used_seeds=[1001], used_panel_shas=[])
    with pytest.raises(ValueError):
        validate_registration(spec, [], used_seeds=[], used_panel_shas=[spec['panel_sha256']])
    later = registration(2)
    later['paired_seeds'] = list(range(2001, 2006)); later['panel_sha256'] = 'b' * 64
    assert validate_registration(later, [spec], used_seeds=[], used_panel_shas=[])
    damaged_history = copy.deepcopy(spec); damaged_history['alpha'] = .05
    with pytest.raises(ValueError):
        validate_registration(later, [damaged_history], used_seeds=[], used_panel_shas=[])


def test_five_seed_interval_matches_tabulated_t_and_rejects_missing_failed_seeds():
    spec = registration(); gains = dict(zip(spec['paired_seeds'], [.02, .03, .04, .05, .06]))
    result = fixed_sample_result(spec, gains)
    # df=4, two-sided alpha=.025: t(.9875)=3.49540593251644.
    half = 3.4954059325164417 * math.sqrt(.00025) / math.sqrt(5)
    assert result['interval'] == pytest.approx([.04 - half, .04 + half])
    assert result['decision'] == 'USEFUL_GAIN_SUPPORTED'
    for bad in [dict(list(gains.items())[:-1]), dict(gains, extra=.1),
                gains | {1001: float('nan')}]:
        with pytest.raises(ValueError): fixed_sample_result(spec, bad)
    later = dict(spec, attempt=2, alpha=allocated_alpha(2))
    assert fixed_sample_result(later, gains)['interval'][0] < result['interval'][0]


def test_decision_retains_meaningful_threshold_and_reports_inconclusive():
    spec = registration()
    for values, expected in [([.01] * 5, 'INCONCLUSIVE'),
                             ([.002, .003, .004, .005, .006], 'USEFUL_GAIN_RULED_OUT'),
                             ([.0, .01, .02, .03, .04], 'INCONCLUSIVE')]:
        assert fixed_sample_result(spec, dict(zip(spec['paired_seeds'], values)))['decision'] == expected
