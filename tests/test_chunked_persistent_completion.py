import pytest
from scripts.run_chunked_persistent_completion import check_partial_evidence


def evidence():
    checks={str(k):dict(overall=dict(relative_l2=.0001),body=dict(relative_l2=.0002),loss_error=.00001) for k in range(4)}
    updates=[dict(update=u,extra_steps=(u-1)%4,seconds=.3) for u in range(1,104)]
    return checks,updates


def test_only_complete_successful_partial_measurements_can_be_reused():
    checks,updates=evidence();assert check_partial_evidence(checks,updates)==pytest.approx({str(k):.3 for k in range(4)})
    with pytest.raises(AssertionError):check_partial_evidence(checks,updates[:-1])
    checks['2']['body']['relative_l2']=.021
    with pytest.raises(AssertionError):check_partial_evidence(checks,updates)


def test_state_recovery_cannot_relax_throughput_or_loss_limits():
    checks,updates=evidence();checks['3']['loss_error']=.0031
    with pytest.raises(AssertionError):check_partial_evidence(checks,updates)
    checks,updates=evidence()
    for row in updates:
        if row['extra_steps']==2:row['seconds']=1.1
    with pytest.raises(AssertionError):check_partial_evidence(checks,updates)
