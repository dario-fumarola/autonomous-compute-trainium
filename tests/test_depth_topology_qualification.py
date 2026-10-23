"""Metadata timing/gate fixtures only."""
import pytest

from scripts.qualify_depth_topology import paired_warm_rate, prerequisite_state


def test_performance_queue_cannot_take_chips_from_scientific_prerequisites():
    done={'status':'COMPLETE_VARIABLE_DEPTH_CAMPAIGN'}
    assert prerequisite_state(None,None)=='WAIT'
    assert prerequisite_state(done,None)=='WAIT'
    assert prerequisite_state({'status':'STOPPED_VARIABLE_DEPTH_CAMPAIGN'},None)=='STOP_PRIMARY_FAILED'
    assert prerequisite_state(done,{'status':'UNEXECUTED_PRIMARY_FAILED'})=='STOP_ANNEX_INCOMPLETE'
    assert prerequisite_state(done,{'status':'COMPLETE_READONLY_DEPTH_DIAGNOSTICS'})=='READY'
    assert prerequisite_state(done,{'status':'UNEXECUTED_NATIVE_TIME_GATE'})=='READY'


def test_pair_completion_rate_counts_actual_overlap_instead_of_summing_rates():
    rows=[dict(warm_updates=100,warm_started_unix=10,warm_ended_unix=20,warm_real_targets=1000),
          dict(warm_updates=100,warm_started_unix=12,warm_ended_unix=32,warm_real_targets=1000)]
    result=paired_warm_rate(rows)
    assert result['wall_seconds']==22 and result['real_targets_per_second']==pytest.approx(2000/22)
    assert result['real_targets_per_second']<1000/10+1000/20
    with pytest.raises(ValueError):paired_warm_rate(rows[:1])
    with pytest.raises(ValueError):paired_warm_rate([dict(rows[0],warm_updates=12),rows[1]])
    with pytest.raises(ValueError):paired_warm_rate([dict(rows[0],warm_ended_unix=float('nan')),rows[1]])
