import pytest
from compute_trainium import execution_isolation as iso


def test_routine_failure_never_cancels_the_other_campaign(monkeypatch):
    monkeypatch.setattr(iso, 'metadata', lambda name: {'Config': {'Labels': {
        iso.MANAGED: '1', iso.OWNER: 'pilot'}}})
    monkeypatch.setattr(iso, 'managed_running', lambda: ['pilot-container', 'important-control'])
    for reason in ['supervisor_timeout', 'compiler_error_in_log_fail_closed', 'container_oom_kill', 'numeric_failure']:
        assert iso.cancellation_targets('pilot-container', 'pilot', reason) == ['pilot-container']
    assert iso.cancellation_targets('pilot-container', 'pilot', 'host_memory_reserve') == ['pilot-container', 'important-control']


def test_wrong_owner_cannot_cancel_any_container(monkeypatch):
    monkeypatch.setattr(iso, 'metadata', lambda name: {'Config': {'Labels': {
        iso.MANAGED: '1', iso.OWNER: 'control'}}})
    with pytest.raises(ValueError, match='owner'):
        iso.cancellation_targets('control-container', 'pilot', 'filesystem_reserve')
