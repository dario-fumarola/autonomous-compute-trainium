import json

import pytest
import torch

from compute_trainium.training import file_sha256
from scripts.run_parcae_optimizer_native import comparison, launch_ready, step_passes


def test_optimizer_queue_requires_pass_and_cannot_start_late():
    passed = {'status': 'PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS'}
    assert not launch_ready(None, False, 1, 2)
    assert not launch_ready(passed, True, 1, 2)
    assert launch_ready(passed, False, 1, 2)
    with pytest.raises(TimeoutError):
        launch_ready(passed, False, 3, 2)
    with pytest.raises(RuntimeError):
        launch_ready({'status': 'STOPPED'}, False, 1, 2)


def test_optimizer_file_comparison_preserves_update_and_replay_units(tmp_path):
    root = tmp_path / 'reference'
    gradient = tmp_path / 'gradient'
    runs = tmp_path / 'runs'
    for path in [root / 'ordinary', gradient / 'ordinary', runs / 'primary', runs / 'replay']:
        path.mkdir(parents=True)
    initial = {'w': torch.ones(2)*100., 'matrix': torch.ones(2)*100.}
    # Real checkpoints include non-parameter RoPE buffers and a tied head alias;
    # independent update vectors deliberately contain named parameters only.
    full_initial = {**initial, 'buffer': torch.zeros(4), 'tied_alias': initial['w']}
    torch.save({'model': full_initial}, gradient / 'ordinary' / 'qualification-state.pt')
    vectors = []
    for update in range(1, 4):
        model = {n: v-update*.01 for n, v in initial.items()}
        record = {'model': model, 'moments': {
            'w': {'m': torch.ones(2)*.2, 'v': torch.ones(2)*.01, 'step': update},
            'matrix': {'m': torch.ones(2)*.2, 'v': torch.ones(2)*.01}}}
        file = root / 'ordinary' / f'cpu-step-{update}.pt'
        torch.save(record, file)
        vectors.append({'checkpoint_sha256': file_sha256(file)})
        torch.save(record, runs / 'primary' / f'native-step-{update}.pt')
    (root / 'report.json').write_text(json.dumps({'records': [{'arm': 'ordinary', 'vectors': vectors}]}))
    record['model'] = {n: v+.001 for n, v in record['model'].items()}
    torch.save(record, runs / 'replay' / 'native-step-3.pt')
    result = comparison(root, gradient, runs, 'ordinary')
    assert all(r['update_comparison']['relative_l2'] == 0 for r in result['updates'])
    assert .09 < result['replay_vs_uninterrupted_update']['relative_l2'] < .11
    assert .09 < result['replay_vs_cpu']['update_comparison']['relative_l2'] < .11
    assert result['replay_vs_cpu']['moment_comparison']['relative_l2'] == 0
    assert all(step_passes(r) for r in result['updates'])
    assert not step_passes(result['replay_vs_cpu'])


def test_large_embedding_update_cannot_hide_a_failed_muon_family():
    row = {key: {'relative_l2': 1e-6} for key in ['update_comparison', 'moment_comparison']}
    row['families'] = {kind: {key: {'relative_l2': 1e-6} for key in ['update_comparison', 'moment_comparison']}
                       for kind in ['adamw', 'muon']}
    assert step_passes(row)
    row['families']['muon']['update_comparison']['relative_l2'] = .03
    assert not step_passes(row)
