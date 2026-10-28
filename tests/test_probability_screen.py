import pytest
from scripts.analyze_probability_screen import paired
from scripts.train_probability_screen import learning_rate

def test_cooldown_applies_to_each_independent_budget():
    for total in (1530,3400):
        arm=dict(depth_schedule=[0]*total,warmup_updates=32,decay_start=int(.9*total))
        values=[learning_rate(u,arm,.001) for u in range(1,total+1)]
        assert values[31]==.001 and values[arm['decay_start']-1]==.001
        assert values[-1]==pytest.approx(.0001)
        assert all(a>=b for a,b in zip(values[arm['decay_start']-1:],values[arm['decay_start']:]))

def test_paired_weighting_and_source_identity():
    left=[dict(id=d,group=d,domain=d,targets=10,loss_sum=10) for d in ('web','math','code')]
    right=[dict(r,loss_sum=10-g) for r,g in zip(left,(1,2,3))]
    result=paired(left,right,seed=9,replicates=20,weighted=True)
    assert result['gain']==pytest.approx(.16)
    assert result['conditional_95_interval']==pytest.approx([.16,.16])
    right[0]['targets']=9
    with pytest.raises(AssertionError):paired(left,right,seed=9,replicates=20,weighted=True)
