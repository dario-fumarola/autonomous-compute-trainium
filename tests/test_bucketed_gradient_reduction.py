import copy
import pytest
import torch

from compute_trainium.bucketed_gradient_reduction import reduce_and_clip


def parameters(scale=1., dtype=torch.float64):
    torch.manual_seed(8401)
    result = [torch.nn.Parameter(torch.randn(*s,dtype=dtype)) for s in ((3,7),(5,),(2,4),(1,))]
    for p in result: p.grad = scale*torch.randn_like(p)
    return result


@pytest.mark.parametrize('scale',[0.,0.001,10.])
@pytest.mark.parametrize('bucket_bytes',[1,80,100000])
def test_complete_mean_clip_and_optimizer_match_per_parameter_reference(scale,bucket_bytes):
    left = parameters(scale); right = copy.deepcopy(left)
    for a,b in zip(left,right): b.grad=a.grad.clone()
    # Rank-specific offsets make a sum distinguishable from a replicated mean.
    remote = [torch.linspace(-.2,.3,p.numel(),dtype=p.dtype).reshape_as(p) for p in left]
    cursor = 0; calls = []
    def reduce_sum(flat):
        nonlocal cursor
        # The remote stream has the same parameter order across any bucketing.
        other = torch.cat([g.reshape(-1) for g in remote])
        flat.add_(other[cursor:cursor+flat.numel()]);cursor+=flat.numel();calls.append(flat.numel())
    for p,g in zip(right,remote): p.grad.add_(g).div_(2)
    expected_norm = torch.nn.utils.clip_grad_norm_(right,1.,error_if_nonfinite=True)
    norm,buckets = reduce_and_clip(left,reduce_sum,2,bucket_bytes=bucket_bytes)
    assert buckets == len(calls) and cursor == sum(p.numel() for p in left)
    torch.testing.assert_close(norm,expected_norm,rtol=1e-12,atol=1e-12)
    for a,b in zip(left,right):torch.testing.assert_close(a.grad,b.grad,rtol=1e-12,atol=1e-12)
    aopt=torch.optim.AdamW(left,lr=.001,betas=(.9,.95),weight_decay=.1,foreach=False)
    bopt=torch.optim.AdamW(right,lr=.001,betas=(.9,.95),weight_decay=.1,foreach=False)
    aopt.step();bopt.step()
    for a,b in zip(left,right):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(aopt.state[a][key],bopt.state[b][key],rtol=1e-12,atol=1e-12)


def test_bucket_views_survive_next_zero_and_backward():
    p = parameters(dtype=torch.float32)
    opt = torch.optim.AdamW(p,lr=.001,foreach=False)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        sum(v.square().sum() for v in p).backward()
        _,n=reduce_and_clip(p,lambda b:None,1,bucket_bytes=100000)
        assert n==1
        opt.step()
    assert all(torch.isfinite(v).all() for v in p)


def test_zero_norm_is_finite_and_preserves_zero_gradients():
    p=parameters(scale=0.)
    norm,_=reduce_and_clip(p,lambda b:None,1,bucket_bytes=80)
    assert norm == 0 and all(torch.count_nonzero(v.grad)==0 for v in p)


def test_missing_duplicate_mixed_and_nonfinite_gradients_fail_closed():
    p=parameters()
    with pytest.raises(ValueError):reduce_and_clip(p+[p[0]],lambda x:None,1)
    p[0].grad=None
    with pytest.raises(ValueError):reduce_and_clip(p,lambda x:None,1)
    p=parameters();q=parameters(dtype=torch.float32)
    with pytest.raises(ValueError):reduce_and_clip(p+q,lambda x:None,1)
    p=parameters();p[0].grad[0,0]=float('nan')
    with pytest.raises(RuntimeError):reduce_and_clip(p,lambda x:None,1)
