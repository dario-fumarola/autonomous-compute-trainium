import pytest
import torch
from torch.nn import functional as F
from compute_trainium.attention_execution_probe import attention


@pytest.mark.parametrize('length,tile',[(7,3),(16,4),(16,16)])
def test_tiled_attention_matches_independent_sdpa_forward_and_all_gradients(length,tile):
    torch.manual_seed(9840)
    values=[torch.randn(2,2,length,8,dtype=torch.float64) for _ in range(3)]
    a=[v.clone().requires_grad_() for v in values];b=[v.clone().requires_grad_() for v in values]
    x=attention(*a,tile=tile);y=F.scaled_dot_product_attention(*b,is_causal=True)
    torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
    x.square().mean().backward();y.square().mean().backward()
    for v,w in zip(a,b):torch.testing.assert_close(v.grad,w.grad,atol=1e-12,rtol=1e-10)
