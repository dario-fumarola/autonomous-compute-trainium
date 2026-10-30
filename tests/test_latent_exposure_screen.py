"""Budget and finite-gate fixtures; no research examples."""
import pytest
from scripts.run_latent_exposure_screen import arm_for,arithmetic_statistics,advancement,trial_identity


def test_every_conditional_role_has_a_safe_worker_identity():
    identities=[trial_identity(k) for k in ('ordinary','pause','latent','ordinary_work')]
    assert len(set(identities))==4
    assert all(set(name)<=set('abcdefghijklmnopqrstuvwxyz0123456789-') for name in identities)
    assert identities[-1].endswith('-ordinary-work')
    with pytest.raises(ValueError):trial_identity('unregistered')


def parent():
    return dict(model_config=dict(dim=16,layers=1,heads=2,kv_heads=2,ffn_dim=32,vocab_size=32,max_seq=512),
        arms={'a':dict(depth_schedule=[2,3],workspace_mode='latent',lr=.001,warmup_updates=64)})


def test_complete_schedule_and_recovery_are_separate():
    a=arm_for(parent(),'a','latent',101,1234)
    assert a['checkpoint_updates']==[101] and a['recovery_checkpoint_updates']==[90]
    assert a['depth_schedule'].count(2)==51 and a['depth_schedule'].count(3)==50
    assert a['real_targets']==1234 and a['nominal_matrix_flops']>0
    assert a==arm_for(parent(),'a','latent',101,1234)


def rows():
    return [dict(index=i,group=str(i),stratum=('2','3','4-6')[i%3],targets=1,seed=123,role=r,
        answer_nll=2.+(.1 if r!='latent' else 0),first_nll=2.+(.05 if r!='latent' else 0),exact_prefix_correct=0.)
        for r in ('latent','ordinary','pause') for i in range(96)]


def test_all_cases_and_controls_required():
    v=rows();a=arithmetic_statistics(v,replicates=10)
    assert a['comparisons']['ordinary']['answer']['mean']==pytest.approx(.1)
    with pytest.raises(AssertionError):arithmetic_statistics(v[:-1],replicates=10)
    with pytest.raises(AssertionError):arithmetic_statistics(v+[v[0]],replicates=10)


def test_conditional_spending_requires_every_margin_and_first_target():
    b={'comparisons':{r:{'comparison':dict(gain=.02,conditional_95_interval=[.01,.03])} for r in ('ordinary','pause')}}
    a=arithmetic_statistics(rows(),replicates=10)
    assert advancement(b,a)
    a['comparisons']['pause']['first']['mean']=-.0001
    assert not advancement(b,a)
    a['comparisons']['pause']['first']['mean']=.05
    b['comparisons']['ordinary']['comparison']['gain']=.009
    assert not advancement(b,a)
