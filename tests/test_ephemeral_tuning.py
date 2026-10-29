import copy
import pytest

from compute_trainium.model import ModelConfig
from scripts.run_ephemeral_tuning import admission_evidence, build_arms, GRID
from scripts.analyze_ephemeral_tuning import select


def test_admission_requires_complete_positive_replication_and_qualification():
    rep = dict(status='COMPLETE_INTERLEAVED_SEED_REPLICATION', reproducibility_screen_pass=True)
    qual = dict(status='PASS_EPHEMERAL_ACTUAL_SHAPE_AND_REPLAY')
    admission_evidence(rep, qual)
    for r, q in [(dict(rep, status='RUNNING'), qual),
                 (dict(rep, reproducibility_screen_pass=False), qual),
                 (dict(rep, reproducibility_screen_pass=1), qual),
                 (rep, dict(status='STOPPED_EPHEMERAL_QUALIFICATION'))]:
        with pytest.raises(AssertionError):
            admission_evidence(r, q)


def test_endpoint_source_schedule_retained_but_graph_and_work_change():
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32,
                    vocab_size=23, max_seq=512)
    old = dict(model_config=c.__dict__, batch=2, arms={
        f'latent-lr{i}': dict(lr=lr, checkpoint_updates=[1957], warmup_updates=64,
                            decay_start=1761, depth_schedule=[2]*1957, audit_depths=[2],
                            real_targets=128025288) for i, lr in enumerate(GRID)})
    original = copy.deepcopy(old)
    arms, points = build_arms(old)
    assert old == original
    assert len(arms) == len(points) == 3
    for i, a in enumerate(arms.values()):
        assert a['lr'] == GRID[i]
        assert a['real_targets'] == 128025288
        assert a['warmup_updates'] == 64 and a['decay_start'] == 1761
        assert a['checkpoint_updates'] == [1957]
        assert a['depth_schedule'] == [0]*1957 and a['audit_depths'] == [0]
        assert a['nominal_matrix_flops'] > 0
    old['arms']['latent-lr0']['checkpoint_updates'] = [100]
    with pytest.raises(AssertionError):
        build_arms(old)


def test_selection_cannot_hide_boundary_or_missing_tuning_trial():
    scores = [dict(arm=str(i), lr=lr, broad_nll=v) for i, (lr, v) in enumerate(zip(GRID, [3., 2., 2.5]))]
    assert select(scores, GRID)['arm'] == '1'
    assert select(scores, GRID)['bracketed']
    scores[2]['broad_nll'] = 1.9
    assert not select(scores, GRID)['bracketed']
    with pytest.raises(AssertionError):
        select(scores[:-1], GRID)
