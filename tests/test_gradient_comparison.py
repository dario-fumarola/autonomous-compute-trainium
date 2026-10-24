import math

import pytest
import torch

from scripts.qualify_trained_gradient import compare_gradients
from compute_trainium.depth_mixture import routed_indices


def test_global_gradient_comparison_has_reference_scale_and_direction():
    reference={'a':torch.tensor([3.,4.]),'b':torch.tensor([0.])}
    result=compare_gradients(reference,reference)
    assert result['relative_l2']==0 and result['reference_norm']==5 and result['cosine']==1
    changed={'a':torch.tensor([3.,3.]),'b':torch.tensor([0.])}
    result=compare_gradients(changed,reference)
    assert result['relative_l2']==.2
    assert result['cosine']==pytest.approx(21/(math.sqrt(18)*5))
    with pytest.raises(ValueError):compare_gradients({'wrong':torch.tensor([1.])},reference)
    with pytest.raises(ValueError):compare_gradients({'a':torch.tensor([float('nan'),4.]),'b':reference['b']},reference)


def test_precision_control_repartitions_exactly_the_same_global_source_windows():
    for cursor in [0,1966080]:
        usual=[i for micro in range(2) for rank in range(32) for i in routed_indices(cursor,micro,rank,32,4,2,271828)]
        smaller=[i for micro in range(8) for rank in range(32) for i in routed_indices(cursor,micro,rank,32,1,8,271828)]
        assert len(set(usual))==len(set(smaller))==256
        assert sorted(usual)==sorted(smaller)==list(range(cursor,cursor+256))
