import pytest

from compute_trainium.scale_pilot_size import profiled_inventory
from compute_trainium.scale_work_ledger import fixed_update_work,total_work,work_matched_updates


def test_long_fits_keep_exact_integer_totals_and_match_the_control_without_overflow():
    candidate,control=[fixed_update_work(row['config'],row['role']) for row in profiled_inventory('medium')]
    assert candidate['owner_optimizer_per_update']>0
    assert candidate['total_per_update']==candidate['neural_per_update']+candidate['owner_optimizer_per_update']
    budget=total_work(candidate['total_per_update'],20000)
    assert budget>2**63 and isinstance(budget,int)
    result=work_matched_updates(candidate['total_per_update'],20000,control['total_per_update'])
    assert 0<=result['unmatched_work']<control['total_per_update']
    assert result['candidate_total']-result['control_total']==result['unmatched_work']
    with pytest.raises(ValueError):total_work(float(candidate['total_per_update']),20000)
