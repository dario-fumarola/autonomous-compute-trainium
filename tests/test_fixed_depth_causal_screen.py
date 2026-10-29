import copy
import pytest

from compute_trainium.model import ModelConfig
from scripts.run_fixed_depth_causal_screen import build_arms, schedule
from scripts.analyze_fixed_depth_causal import decisions
from scripts.readout_fixed_depth_causal import indices_for_worker


def test_schedule_only_change_and_independent_work():
    c = ModelConfig(dim=512,layers=6,heads=8,kv_heads=8,ffn_dim=1376,vocab_size=32000,max_seq=512)
    old = dict(model_config=c.__dict__,batch=2,arms={f'latent-lr{i}':dict(lr=lr,checkpoint_updates=[1957],seed=9901)
               for i,lr in enumerate([.0003,.001,.003])})
    saved = copy.deepcopy(old)
    arms,points = build_arms(old)
    assert old == saved and schedule() == [1]*1957
    projection = 6*512*(8*512**2+6*512*1376)*2
    attention = 6*4*512**2*512*3
    head = 2*512*512*32000
    work = 1957*64*2*3*(projection+attention+head)
    assert work == 49_069_855_327_911_936
    for a in arms.values():
        assert a['depth_schedule'] == [1]*1957
        assert a['audit_depths'] == [1]
        assert a['nominal_matrix_flops'] == work
        assert a['seed'] == 9901
    assert all([v['depth'] for v in p['variants']] == [1] for p in points)


@pytest.mark.parametrize('failure',['fixed_over_mixed','fixed_over_strong_ordinary'])
def test_either_gate_blocks_future_expense(failure):
    rows={k:dict(comparison=dict(gain=.03,conditional_95_interval=[.01,.04])) for k in ('fixed_over_mixed','fixed_over_strong_ordinary')}
    assert all(decisions(rows,True).values())
    rows[failure]['comparison']['gain'] = 0.
    assert not all(decisions(rows,True).values())


def test_boundary_or_nonpositive_interval_cannot_pass():
    rows={k:dict(comparison=dict(gain=.03,conditional_95_interval=[.01,.04])) for k in ('fixed_over_mixed','fixed_over_strong_ordinary')}
    assert not any(decisions(rows,False).values())
    rows['fixed_over_mixed']['comparison']['conditional_95_interval'][0] = 0.
    assert not decisions(rows,True)['fixed_over_mixed']


def test_exact_readout_complete_partition():
    indices=[i for rank in range(4) for batch in indices_for_worker(rank,4) for i in batch]
    assert sorted(indices) == list(range(384))
