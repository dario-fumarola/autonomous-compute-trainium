import pytest
import torch

from scripts.run_parcae_fp16_native import compare, launch_ready


def test_fp16_queue_needs_both_native_gates_and_cannot_start_late():
    gradient = {'status': 'PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS'}
    optimizer = {'status': 'PASS_PAIRED_NATIVE_PARCAE_OPTIMIZER_AND_RECOVERY'}
    assert not launch_ready(gradient, None, False, 1, 2)
    assert not launch_ready(None, optimizer, False, 1, 2)
    assert not launch_ready(gradient, optimizer, True, 1, 2)
    assert launch_ready(gradient, optimizer, False, 1, 2)
    with pytest.raises(TimeoutError):
        launch_ready(gradient, optimizer, False, 3, 2)
    with pytest.raises(RuntimeError):
        launch_ready({'status': 'STOPPED'}, optimizer, False, 1, 2)


def test_fp16_gradient_gate_does_not_hide_body_error_behind_head(tmp_path):
    values = {'transformer.wte.weight': torch.ones(2)*100.,
              'transformer.core_block.0.attn.c_q.weight': torch.ones(2)*.001}
    expected = {'loss': 3., 'gradients': values}
    actual = {'loss': 3., 'gradients': {k: v.clone() for k,v in values.items()}}
    actual['gradients']['transformer.core_block.0.attn.c_q.weight'] *= 1.2
    torch.save(expected, tmp_path / 'reference.pt')
    torch.save(actual, tmp_path / 'actual.pt')
    row = compare(tmp_path / 'reference.pt', tmp_path / 'actual.pt')
    assert row['global_gradient']['relative_l2'] < .00001
    assert row['gradient_groups']['transformer_body']['relative_l2'] > .19
    assert not row['passes_registered_gates']
