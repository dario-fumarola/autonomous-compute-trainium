import copy
import pytest
from compute_trainium.bucketed_benchmark import ORDER,summarize


def fixture(times):
    return [dict(trial=n,updates=[dict(update=i,seconds=(t if i>=4 else 1000.)) for i in range(1,104)]) for n,t in zip(ORDER,times)]


def test_each_order_must_clear_speed_gate_and_compile_time_is_excluded():
    x=summarize(fixture([1.,.9,.94,1.]))
    assert x['speed_gate_passed'] and x['automatic_promotion'] is False
    assert x['order_specific_ratios']==pytest.approx([.9,.94])
    assert not summarize(fixture([1.,.5,.96,1.]))['speed_gate_passed']


def test_missing_duplicate_reordered_and_invalid_timings_rejected():
    x=fixture([1.,1.,1.,1.])
    for bad in (x[:-1],list(reversed(x))):
        with pytest.raises(ValueError):summarize(bad)
    bad=copy.deepcopy(x);bad[0]['updates'].pop()
    with pytest.raises(ValueError):summarize(bad)
    bad=copy.deepcopy(x);bad[1]['updates'][5]['seconds']=float('nan')
    with pytest.raises(ValueError):summarize(bad)
