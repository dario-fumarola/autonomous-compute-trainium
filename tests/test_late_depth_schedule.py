"""Integer software fixtures, never scientific corpus examples."""
import numpy as np
import pytest

from compute_trainium.late_depth_schedule import (switch_update,staged_work,
    matched_late_endpoint,staged_depths,validate_staged_depths)


def test_exact_switch_and_all_steps_receive_gradients():
    values=staged_depths(12,microbatches=2,world=3)
    assert switch_update(12)==9 and values.shape==(13,2,3,2)
    assert np.all(values[:9]==1) and np.all(values[9:]==2)
    assert validate_staged_depths(values,12)
    values[9,0,0,1]=1
    with pytest.raises(ValueError): validate_staged_depths(values,12)


def test_integer_endpoint_matches_exhaustive_independent_work_sum():
    for budget in range(49,500):
        result=matched_late_endpoint(budget,7,13,maximum_updates=100)
        possible=[n for n in range(4,100) if sum([7]*(3*n//4)+[13]*(n-3*n//4))<=budget]
        assert result['total_updates']==max(possible)
        assert result['training_matrix_ops']<=budget<staged_work(result['total_updates']+1,7,13)


def test_allocation_is_prefix_fixed_across_checkpoint_segments():
    original=staged_depths(576)
    pieces=np.concatenate([original[:121],original[121:432],original[432:576],original[576:]])
    assert np.array_equal(original,pieces)
    # Recomputing a schedule on a completed segment would be a different plan.
    assert not np.array_equal(original[:121],staged_depths(120))


def test_insufficient_budget_and_unprovisioned_endpoint_fail():
    with pytest.raises(ValueError): matched_late_endpoint(1,7,13,maximum_updates=100)
    with pytest.raises(ValueError): matched_late_endpoint(10000,7,13,maximum_updates=100)
