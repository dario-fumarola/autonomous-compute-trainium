import pytest
from scripts.run_interleaved_tuning import updates_for_targets
from scripts.train_interleaved_screen import learning_rate

class Sources:
    windows=12
    def target_count(self,start,count):
        assert start==0
        return sum([3,4,2,4,4,1,4,3,4,4,4,2][:count])

@pytest.mark.parametrize('targets,expected',[(1,1),(7,1),(8,2),(13,2),(14,3),(39,6)])
def test_real_target_budget_rounds_once_to_global_update(targets,expected):
    data=Sources();u=updates_for_targets(data,targets,2)
    assert u==expected and data.target_count(0,2*u)>=targets
    if u>1:assert data.target_count(0,2*(u-1))<targets


def test_source_exhaustion_and_complete_cooldown():
    with pytest.raises(ValueError):updates_for_targets(Sources(),40,2)
    arm=dict(warmup_updates=64,decay_start=1760,depth_schedule=[2]*1956)
    assert learning_rate(1,arm,.003)==pytest.approx(.003/64)
    assert learning_rate(64,arm,.003)==.003
    assert learning_rate(1760,arm,.003)==.003
    assert learning_rate(1956,arm,.003)==pytest.approx(.0003)
