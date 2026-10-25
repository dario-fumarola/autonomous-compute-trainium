import pytest
from scripts.collect_low_depth_resources import union_seconds


def test_concurrent_intervals_are_not_charged_twice_and_gaps_remain():
    assert union_seconds([(0,5),(2,7),(7,8),(10,12),(1,3)])==10
    assert union_seconds([])==0
    with pytest.raises(ValueError):union_seconds([(2,1)])
