import numpy as np
import pytest

from compute_trainium.future_span_control_budget import control_branch


def test_control_pays_inherited_prefix_and_owns_a_complete_cooldown():
    # Integer bookkeeping fixtures, never model inputs or research examples.
    work = np.array([10] * 13, dtype=np.int64)
    prefix = np.array([[3*i, i, i] for i in range(27)], dtype=np.int64)
    result = control_branch(work, prefix, 125, origin_update=4,
                            global_windows=2, warmup_targets=10)
    assert result['total_updates'] == 12
    assert result['cooldown_start_update'] == 6
    assert result['matched_work']['executed_matmuls'] == 120
    assert result['matched_work']['unused_budget_matmuls'] == 5
    assert result['inherited_matrix_ops_charged'] == 40
    assert result['additional_branch_matrix_ops'] == 80
    assert result['final_source']['real_targets'] == 120
    assert not result['training_launched'] and not result['checkpoint_adopted']


def test_control_cannot_rewind_a_later_or_annealed_schedule():
    work = np.array([10] * 13, dtype=np.int64)
    prefix = np.array([[3*i, i, i] for i in range(27)], dtype=np.int64)
    with pytest.raises(ValueError, match='later'):
        control_branch(work, prefix, 125, origin_update=7,
                       global_windows=2, warmup_targets=10)
    with pytest.raises(ValueError, match='following'):
        control_branch(work[:12], prefix, 120, origin_update=4,
                       global_windows=2, warmup_targets=10)
    with pytest.raises(ValueError, match='warmup'):
        control_branch(work, prefix, 125, origin_update=4,
                       global_windows=2, warmup_targets=100)
    with pytest.raises(ValueError, match='Provisioned'):
        control_branch(work, prefix[:10], 125, origin_update=4,
                       global_windows=2, warmup_targets=10)
