"""Numerical fixtures only; no generated research examples."""
import numpy as np
import pytest

from compute_trainium.prefix_value_probe import lag_features, group_fold, crossfit_probe


def test_current_and_future_losses_cannot_change_current_features():
    values = np.arange(12, dtype=float)
    frames = [dict(offset=0, length=12)]
    original = lag_features(values, frames, context=4)
    for current in range(12):
        changed = values.copy(); changed[current:] += 100
        assert np.array_equal(lag_features(changed, frames, context=4)[:current+1], original[:current+1])


def test_window_and_document_history_resets():
    frames = [dict(offset=0, length=6), dict(offset=6, length=3)]
    features = lag_features(np.ones(9)*7, frames, context=4)
    assert np.array_equal(features[[0, 4, 6]], np.zeros((3, 5)))
    assert np.array_equal(features[[1, 5, 7], 0], np.array([7., 7., 7.]))


def test_groups_remain_together_and_bad_coverage_fails():
    assert group_fold('same-host') == group_fold('same-host')
    with pytest.raises(ValueError): lag_features(np.ones(3), [dict(offset=1, length=2)])
    with pytest.raises(ValueError): group_fold('')


def fixture():
    groups = {}
    for i in range(1000):
        group = 'fixture-'+str(i); groups.setdefault(group_fold(group), group)
        if len(groups) == 5: break
    frames = []; values = []; cursor = 0
    for fold, group in sorted(groups.items()):
        for domain in ['web', 'math', 'code']:
            frames.append(dict(offset=cursor, length=8, group=group, domain=domain))
            values.extend(np.linspace(-1., 1., 8)); cursor += 8
    x = np.array(values)[:, None]
    losses = np.stack((2.+.2*x[:, 0], 2.-.2*x[:, 0], np.full(len(x), 2.)))
    return x, losses, frames


def test_heldout_labels_cannot_change_heldout_predictions():
    x, losses, frames = fixture(); original = crossfit_probe(x, losses, frames)
    heldout = original['folds'] == 0
    changed = losses.copy(); changed[:, heldout] += np.array([[1.], [3.], [5.]])
    other = crossfit_probe(x, changed, frames)
    assert np.array_equal(original['predicted_differences'][heldout], other['predicted_differences'][heldout])


def test_crossfit_recovers_known_linear_differences_without_target_features():
    x, losses, frames = fixture(); result = crossfit_probe(x, losses, frames, ridge=1e-9)
    expected = (losses[:2]-losses[2]).T
    assert np.allclose(result['predicted_differences'][:, :2], expected, atol=1e-8)
    assert np.array_equal(result['choices'], np.argmin(losses, axis=0))
