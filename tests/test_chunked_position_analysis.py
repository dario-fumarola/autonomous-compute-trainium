"""Numerical loss arrays only; not research examples."""
import copy

import numpy as np
import pytest

from compute_trainium.chunked_position_analysis import position_strata, make_rows, validate, compare


def rows(values=None):
    metadata = [dict(id=str(i), group=str(i//2), domain=('web','math','code')[i//128]) for i in range(384)]
    return make_rows(np.ones((384,512)) if values is None else values, metadata, list(range(384)))


def test_partition_has_exact_boundary_alignment_and_no_dropped_targets():
    p = position_strata()
    assert [len(v) for v in p.values()] == [127,4,24,72,285]
    assert sorted(i for v in p.values() for i in v) == list(range(512))
    assert p['slot_prediction'] == [127,255,383,511]
    assert p['after_1_to_8'][:8] == list(range(128,136))
    assert p['after_33_to_127'][-1] == 510
    with pytest.raises(ValueError): position_strata(256)


def test_boundary_only_gain_has_small_exact_whole_panel_contribution():
    original = rows(); v = np.ones((384,512)); v[:,[127,255,383,511]] = .5
    result = compare(rows(v), original, replicates=20)
    assert result['full']['broad']['gain'] == pytest.approx(.5*4/512)
    assert result['strata']['slot_prediction']['broad']['gain'] == pytest.approx(.5)
    assert result['strata']['after_1_to_8']['broad']['gain'] == 0
    assert result['strata']['slot_prediction']['contribution_to_broad_gain'] == pytest.approx(.5*4/512)


def test_weighted_domain_signs_and_complete_pairing():
    original = rows(); v = np.ones((384,512)); v[:128] += .2; v[128:256] -= .1
    result = compare(rows(v), original, replicates=20)
    assert result['full']['broad']['gain'] == pytest.approx(-.1)
    bad = copy.deepcopy(original); bad[0]['group'] = 'unpaired'
    with pytest.raises(AssertionError): compare(bad, original, replicates=20)


def test_nonfinite_losses_duplicate_rows_and_inconsistent_sums_are_rejected():
    v = np.ones((384,512)); v[0,0] = np.nan
    with pytest.raises(AssertionError): rows(v)
    a = rows(); a[0]['strata']['slot_prediction']['loss_sum'] += 1
    with pytest.raises(AssertionError): validate(a)
    with pytest.raises(AssertionError): validate(rows()[:-1])
    a = rows(); a[0]['index'] = 1
    with pytest.raises(AssertionError): validate(a)
