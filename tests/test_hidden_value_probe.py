"""Numerical causality/statistical software fixtures, never research examples."""
import numpy as np
import torch

from compute_trainium.hidden_value_probe import (prefix_features, fit_readouts,
    predict_readouts, calibrated_mask, allocation_curve, matched_curve_nll)
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_stages_candidate import ParcaeStagesCandidate


def test_future_source_and_current_target_cannot_change_prefix_features():
    torch.manual_seed(41)
    config = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=32,
        prelude=1, core=2, coda=1, max_seq=16)
    model = ParcaeStagesCandidate(config).eval()
    with torch.no_grad():
        for p in model.parameters(): p.add_(torch.randn_like(p)*.03)
        tokens = torch.randint(0, 32, (1, 12)); state = torch.randn(1, 12, 32)
        def features(values):
            source = model.encode(values)
            hidden = model.refine(state, source, values)
            return prefix_features(hidden[0], model.decode(hidden, values)[0], context=16)
        before = features(tokens)
        changed = tokens.clone(); changed[:, 7:] = (changed[:, 7:]+1) % 32
        # Feature at position6 predicts token7, which we have changed.
        torch.testing.assert_close(before[:7], features(changed)[:7], rtol=0, atol=0)


def test_uniform_distribution_entropy_and_position_are_prefix_only():
    features = prefix_features(torch.zeros(4, 2), torch.zeros(4, 8), context=8)
    torch.testing.assert_close(features[:, 2], torch.full((4,), float(np.log(8))))
    assert torch.equal(features[:, 3], torch.zeros(4))
    torch.testing.assert_close(features[:, 4], torch.arange(4)/7)


def test_train_threshold_does_not_rank_future_heldout_positions():
    calibration = np.arange(100.)
    scores = np.arange(10.)+45
    original, threshold = calibrated_mask(calibration, scores, .5)
    changed = scores.copy(); changed[6:] = -999
    result, other_threshold = calibrated_mask(calibration, changed, .5)
    assert threshold == other_threshold
    assert np.array_equal(original[:6], result[:6])


def test_readout_predictions_use_no_heldout_labels_and_fit_linear_signal():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(90, 3)).astype(np.float32)
    domains = np.tile([1, 2, 3], 30)
    gain = .2*x[:, 0]-.1*x[:, 2]+.03
    fitted = fit_readouts(x, gain, domains, ridge=1e-8, epochs=1, batch_size=32)
    predictions = predict_readouts(fitted, x)
    assert np.allclose(predictions['linear'], gain, atol=1e-6)
    losses = np.stack((np.full(90, 2.), 2.-gain))
    before = allocation_curve(predictions['linear'], predictions['linear'], losses, domains)
    changed = losses.copy(); changed[1] += .2
    after = allocation_curve(predictions['linear'], predictions['linear'], changed, domains)
    assert [r['threshold'] for r in before] == [r['threshold'] for r in after]
    assert [r['selected_targets'] for r in before] == [r['selected_targets'] for r in after]
    assert before[3]['gain_over_same_fraction_random'] > .04


def test_cost_interpolation_uses_fractions_not_favorable_losses():
    rows = [dict(realized_token_fraction=q, nll=v) for q, v in [(0., 3.), (.4, 2.9), (1., 2.8)]]
    assert abs(matched_curve_nll(rows, .2)-2.95) < 1e-12
    rows[-1]['nll'] = 0.
    assert abs(matched_curve_nll(rows, .2)-2.95) < 1e-12
