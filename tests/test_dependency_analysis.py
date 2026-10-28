import pytest
from compute_trainium.dependency_analysis import interval


def test_paired_interval_weights_real_targets_and_resamples_whole_groups():
    rows=[dict(group='a',real_targets=1),dict(group='a',real_targets=1),dict(group='b',real_targets=6)]
    result=interval(rows,[1.,3.,0.],draws=1000);assert result['mean']==pytest.approx(.5);assert result['real_targets']==8 and result['source_groups']==2
    constant=interval(rows,[2.,2.,2.],draws=1000);assert constant['mean']==2. and constant['conditional_source_group_95_interval']==[2.,2.]
