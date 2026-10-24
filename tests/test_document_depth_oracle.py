import itertools
import numpy as np
import pytest

from compute_trainium.document_depth_oracle import weighted_document_losses, hindsight_point


def test_oracle_separable_solution_equals_exhaustive_document_assignments():
    loss = np.array([[.2, .7, .4], [.15, .4, .5], [.16, .3, .3]])
    cost = np.array([[.1, .3, .2], [.2, .6, .4], [.3, .9, .6]])
    previous_cost = float('inf')
    for penalty in [0., .01, .1, 1., 10.]:
        r = hindsight_point(loss, cost, penalty)
        brute = min(sum(loss[c, j]+penalty*cost[c, j] for j, c in enumerate(choices))
                    for choices in itertools.product(range(3), repeat=3))
        assert r['weighted_nll']+penalty*r['normalized_inference_work'] == pytest.approx(brute)
        assert r['normalized_inference_work'] <= previous_cost+1e-12
        previous_cost = r['normalized_inference_work']


def test_document_weighting_counts_all_tokens_before_domain_mixture():
    scores = [{'domain': 'web', 'tokens': 1, 'nll_sum': 10.},
              {'domain': 'web', 'tokens': 99, 'nll_sum': 99.},
              {'domain': 'code', 'tokens': 5, 'nll_sum': 10.}]
    values = weighted_document_losses(scores, {'web': .6, 'code': .4})
    assert values.sum() == pytest.approx(.6*109/100+.4*10/5)
    with pytest.raises(ValueError):
        weighted_document_losses(scores, {'web': 1.})


def test_oracle_rejects_invalid_scores_and_breaks_ties_toward_lower_depth():
    r = hindsight_point([[1., 2.], [1., 2.]], [[1., 1.], [2., 2.]], 0.)
    assert r['choice'].tolist() == [0, 0]
    with pytest.raises(ValueError):
        hindsight_point([[float('nan')]], [[1.]], 0)
