import math

import pytest
import torch

from scripts.qualify_trained_gradient import compare_gradients


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
