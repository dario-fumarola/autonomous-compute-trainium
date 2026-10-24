import pytest

from scripts.run_low_depth_tuning import choose_rates


def test_rate_selection_is_independent_per_arm_and_ties_choose_lower_rate():
    rows=[]
    for arm in ['ordinary','fixed1','poisson2']:
        for lr,nll in [(.002,3.),(.004,2.9),(.008,2.9)]:
            rows.append({'arm':arm,'peak_lr':lr,'status':'COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE',
                         'metrics':{'weighted_nll':nll}})
    selected=choose_rates(rows)
    assert all(r['peak_lr']==.004 for r in selected.values())
    with pytest.raises(ValueError):choose_rates([r for r in rows if r['arm']!='poisson2'])
