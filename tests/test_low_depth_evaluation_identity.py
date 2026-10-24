import pytest

from scripts.evaluate_low_depth_cpu import window_state_key


def test_eval_noise_is_paired_by_original_document_window():
    assert window_state_key('document',2)==window_state_key('document',2)
    assert window_state_key('document',2)!=window_state_key('document',3)
    assert window_state_key('document',2)!=window_state_key('another',2)
    assert 0<=window_state_key('document',2)<2**63-1
    with pytest.raises(ValueError):window_state_key('',0)
