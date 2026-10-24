import copy

import pytest
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.parcae_update_qualification import (
    assign_gradient, compare_moments, compare_update, configure_step, named_moments,
)


def test_update_error_is_measured_against_step_not_large_weights():
    before = {'weight': torch.full((2,), 1000.)}
    expected = {'weight': before['weight'] + .1}
    actual = {'weight': before['weight'] + .11}
    row = compare_update(before, expected, before, actual)
    assert .09 < row['relative_l2'] < .11
    with pytest.raises(ValueError, match='inventory'):
        compare_update({}, expected, before, actual)


def test_moments_include_step_and_all_named_parameters():
    config = ParcaeCandidateConfig(dim=32, heads=2, ffn_dim=64, vocab_size=128, max_seq=8)
    model = ParcaeCandidate(config)
    optimizer = ParcaeOptimizerCandidate(model, gate_policy='release_muon')
    gradients = {n: torch.ones_like(p) * .001 for n, p in model.named_parameters()}
    saved = {n: g.clone() for n, g in gradients.items()}
    configure_step(optimizer, 0)
    assign_gradient(model, gradients, .75)
    optimizer.step()
    assert all(torch.equal(saved[n], g) for n, g in gradients.items())
    moments = named_moments(model, optimizer)
    assert set(moments) == set(gradients)
    assert compare_moments(moments, moments)['relative_l2'] == 0
    damaged = copy.deepcopy(moments)
    key = next(k for k, v in damaged.items() if 'step' in v)
    damaged[key]['step'] += 1
    with pytest.raises(ValueError, match='scalar step'):
        compare_moments(moments, damaged)
    with pytest.raises(ValueError, match='names'):
        assign_gradient(model, {}, 1.)
