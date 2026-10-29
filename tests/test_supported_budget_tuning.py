import copy
from collections import Counter

import pytest

from compute_trainium.model import ModelConfig
from scripts.run_supported_budget_tuning import build_arms, schedule, GRID
from scripts.analyze_supported_budget import gates
from scripts.check_supported_budget_consistency import consistency_decisions
from scripts.readout_supported_budget import indices_for_worker


def test_supported_source_schedule_and_dispatched_work():
    c = ModelConfig(dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32, vocab_size=23, max_seq=512)
    old = dict(model_config=c.__dict__, batch=2, arms={f'latent-lr{i}': dict(lr=lr,
        checkpoint_updates=[1957], real_targets=128025288, warmup_updates=64, decay_start=1761)
        for i, lr in enumerate(GRID)})
    original = copy.deepcopy(old)
    arms, points = build_arms(old)
    assert old == original
    assert schedule() == schedule()
    assert schedule()[:5] == [0, 12, 13, 24, 25]
    assert Counter(schedule()) == {0: 653, 12: 326, 13: 326, 24: 326, 25: 326}
    # Independently reconstruct every dense body, head and backward charge.
    body = c.flops(0, 512, 2)['forward_matmul_flops']
    head = 2*2*512*c.dim*c.vocab_size
    expanded1 = c.flops(0, 1024, 2)['forward_matmul_flops']
    expanded2 = c.flops(0, 1536, 2)['forward_matmul_flops']
    expected = 64*3*(1957*(body+head)+326*(3+4)*expanded1+326*(5+6)*expanded2)
    for a, point in zip(arms.values(), points):
        assert a['nominal_matrix_flops'] == expected
        assert a['real_targets'] == 128025288 and a['checkpoint_updates'] == [1957]
        assert a['warmup_updates'] == 64 and a['decay_start'] == 1761
        assert a['audit_depths'] == [0, 12, 13, 24, 25]
        assert [v['depth'] for v in point['variants']] == [0, 13, 25]
    old['arms']['latent-lr0']['checkpoint_updates'] = [100]
    with pytest.raises(AssertionError):
        build_arms(old)


def test_advancement_cannot_be_manufactured_by_weakening_shallow_exit():
    labels = ['two_over_one', 'two_over_fixed_one', 'two_over_strong_ordinary', 'zero_over_same_shape_ordinary']
    good = {k: dict(comparison=dict(gain=.02, conditional_95_interval=[.01, .03])) for k in labels}
    assert all(gates(good, True, .01, .005).values())
    for label in labels:
        altered = copy.deepcopy(good)
        altered[label]['comparison']['gain'] = -.006 if label == labels[-1] else .009
        assert not all(gates(altered, True, .01, .005).values())
    altered = copy.deepcopy(good)
    altered['two_over_one']['comparison']['conditional_95_interval'][0] = 0
    assert not all(gates(altered, True, .01, .005).values())
    assert not all(gates(good, False, .01, .005).values())


def test_consistency_checks_both_budgets_all_domains_and_absolute_shift():
    limits = dict(maximum_absolute_domain_nll_shift=.01, maximum_domain_kl=.01, maximum_domain_k16_k32_kl=.0001)
    rows = [dict(budget=b, domain=d, round=k, nll=3., exact_nll=3., kl_to_sequential=0., k16_k32_kl=0.)
            for b in (1, 2) for d in ('web', 'math', 'code') for k in (3 if b == 1 else 5, 32) for _ in range(2)]
    assert all(r['passed'] for r in consistency_decisions(rows, limits).values())
    for key, value in [('nll', 2.97), ('kl_to_sequential', .03)]:
        bad = copy.deepcopy(rows)
        next(r for r in bad if r['budget'] == 2 and r['domain'] == 'math' and r['round'] == 5)[key] = value
        assert not consistency_decisions(bad, limits)['2-math']['passed']
    with pytest.raises(AssertionError):
        consistency_decisions(rows[:-1], limits)
    rows[-1]['k16_k32_kl'] = float('nan')
    with pytest.raises(AssertionError):
        consistency_decisions(rows, limits)


def test_exact_workers_cover_each_view_once_without_domain_reweighting():
    batches = [x for rank in range(4) for x in indices_for_worker(rank, 4)]
    assert all(len(b) == 2 for b in batches)
    assert sorted(i for b in batches for i in b) == list(range(384))
    with pytest.raises(ValueError):
        indices_for_worker(4, 4)
