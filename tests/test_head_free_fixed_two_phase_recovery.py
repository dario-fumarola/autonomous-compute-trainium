import pytest

from scripts.run_head_free_fixed_two_phase_recovery import phase_name
from compute_trainium.owned_native_phase import create_worker


def test_every_planned_model_phase_reaches_actual_factory_precondition(tmp_path):
    for trial in ['half_a','half_b']:
        for mode in ['warm','replay']:
            name=phase_name(trial,mode)
            # Valid ID reaches private-environment lookup; no Docker call occurs.
            with pytest.raises(FileNotFoundError):create_worker(tmp_path,name,0)
    with pytest.raises(ValueError,match='safe phase identity'):
        create_worker(tmp_path,'head-free-half_a-warm',0)


def test_unsafe_mode_cannot_enter_a_worker_identity():
    with pytest.raises(ValueError):phase_name('half_a','warm/bad')
