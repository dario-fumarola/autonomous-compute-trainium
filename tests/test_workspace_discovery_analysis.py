import numpy as np
import pytest
from compute_trainium.workspace_discovery_analysis import clustered_interval


def test_group_bootstrap_preserves_pairing_and_is_invariant_to_duplicate_rows():
    values=[.1,.3,.2,.4,.7,.5];groups=['a','b','c','d','e','f'];domains=[0,0,1,1,2,2]
    a=clustered_interval(values,groups,domains,draws=500,seed=41)
    b=clustered_interval(values*2,groups*2,domains*2,draws=500,seed=41)
    assert a['source_groups']==b['source_groups']==6 and b['anchors']==12
    assert a['mean']==pytest.approx(.6*.2+.2*.3+.2*.6)
    np.testing.assert_allclose(a['conditional_source_group_95_interval'],b['conditional_source_group_95_interval'])
    assert 'training-seed' in a['warning']
