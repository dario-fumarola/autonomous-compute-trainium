"""Registration/accounting checks; numerical fixtures are not source examples."""
from collections import Counter
import copy
import json
import time

import pytest

from compute_trainium.model import ModelConfig
from scripts.run_depth_causal_screen import build_arms, schedule
from scripts.run_supported_budget_tuning import schedule as prior_schedule
from scripts.analyze_depth_causal import gates, verify
from scripts.readout_depth_causal import indices_for_worker


def test_identical_source_budget_assignment_and_independent_work_count():
    steps = schedule()
    assert Counter(steps) == {0: 653, 1: 652, 2: 652}
    assert steps == [{0: 0, 12: 1, 13: 1, 24: 2, 25: 2}[k] for k in prior_schedule()]
    c = ModelConfig(dim=512, layers=6, heads=8, kv_heads=8, ffn_dim=1376, vocab_size=32000, max_seq=512)
    old = dict(model_config=c.__dict__, batch=2, arms={f'latent-lr{i}': dict(lr=lr, checkpoint_updates=[1957])
                 for i, lr in enumerate([.0003, .001, .003])})
    original = copy.deepcopy(old)
    arms, points = build_arms(old)
    assert old == original
    # Independent arithmetic: projection matrices once per stage; each stage
    # attends a dense n-by-(stage+1)n rectangle; only the final vocabulary head.
    total = 0
    for budget, count in Counter(steps).items():
        kv = c.dim*c.kv_heads//c.heads
        projection = c.layers*512*(4*c.dim*c.dim+4*c.dim*kv+6*c.dim*c.ffn_dim)*(budget+1)
        attention = c.layers*4*512*512*c.dim*sum(range(1, budget+2))
        head = 2*512*c.dim*c.vocab_size
        total += count*64*2*3*(projection+attention+head)
    assert total == 49_866_412_847_529_984
    assert len(arms) == len(points) == 3
    for arm in arms.values():
        assert arm['depth_schedule'] == steps
        assert arm['audit_depths'] == [0, 1, 2]
        assert arm['nominal_matrix_flops'] == total
    assert all([v['depth'] for v in p['variants']] == [0, 1, 2] for p in points)


def comparisons():
    return {k: dict(comparison=dict(gain=.02, conditional_95_interval=[.015, .025]))
            for k in ['two_over_one', 'two_over_fixed_one', 'two_over_strong_ordinary', 'zero_over_same_shape_ordinary']}


@pytest.mark.parametrize('failure', ['two_over_one', 'two_over_fixed_one', 'two_over_strong_ordinary', 'zero_over_same_shape_ordinary'])
def test_each_original_gate_can_independently_block_advancement(failure):
    rows = comparisons()
    assert all(gates(rows, True, .01, .005).values())
    rows[failure]['comparison']['gain'] = -.006
    assert not all(gates(rows, True, .01, .005).values())


def test_unbracketed_lr_cannot_advance():
    assert not all(gates(comparisons(), False, .01, .005).values())


def test_readout_covers_every_window_exactly_once():
    indices = [i for rank in range(4) for batch in indices_for_worker(rank, 4) for i in batch]
    assert sorted(indices) == list(range(384))


def test_missing_or_failed_endpoint_audit_blocks_all_scoring(tmp_path):
    p = dict(identity='screen', absolute_deadline_unix=time.time()+60, source_sha256={},
             comparison_root_sha256={}, comparison_repo_sha256={}, arms={'a': {}, 'b': {}})
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(p))
    for arm in p['arms']:
        audit = tmp_path/'screen'/('audit-'+arm)
        audit.mkdir(parents=True)
        (audit/'report.json').write_text(json.dumps(dict(status='PASS_INTERLEAVED_SCREEN_MATURE_AUDITS')))
    assert verify(tmp_path, path)['identity'] == 'screen'
    (tmp_path/'screen/audit-b/report.json').write_text(json.dumps(dict(status='FAIL_NUMERICAL')))
    with pytest.raises(AssertionError):
        verify(tmp_path, path)
