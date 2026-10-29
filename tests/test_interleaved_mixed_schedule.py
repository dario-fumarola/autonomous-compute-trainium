import random
import pytest
from scripts.run_interleaved_mixed_tuning import mixed_schedule


def test_mixed_schedule_is_balanced_reproducible_and_does_not_change_training_rng():
    random.seed(125); state=random.getstate()
    a=mixed_schedule(1957)
    assert random.getstate()==state
    assert a==mixed_schedule(1957) and a!=mixed_schedule(1957,9918)
    assert a.count(2)==979 and a.count(3)==978
    assert len(a)==1957


@pytest.mark.parametrize('updates',[0,1,True,2.5])
def test_invalid_mixed_schedule_length(updates):
    with pytest.raises(ValueError):mixed_schedule(updates)
