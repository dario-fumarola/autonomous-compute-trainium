"""Numerical/source-index fixtures only, never research examples."""
import copy

import pytest
import torch

from compute_trainium.depth_mixture import rank_depths, routed_indices
from compute_trainium.depth_topology import resize_depth_signature


def signature():
    return dict(world=32,batch=4,accumulation=2,depth_policy='mixture',
        rank_depths=list(rank_depths(32,'mixture')),routing='splitmix64_fisher_yates_v1',
        routing_seed=271828,corpus_sha256='source',packing_sha256='packing',optimizer='fused')


def assignment(s,cursor):
    return {i:depth for micro in range(s['accumulation']) for rank,depth in enumerate(s['rank_depths'])
            for i in routed_indices(cursor,micro,rank,s['world'],s['batch'],s['accumulation'],s['routing_seed'])}


def test_resize_preserves_every_source_depth_and_does_not_change_other_settings():
    old=signature();new=resize_depth_signature(old,world=16,batch=8)
    for cursor in [0,256,512,4096]:assert assignment(old,cursor)==assignment(new,cursor)
    assert {k:v for k,v in new.items() if k not in ['world','batch','rank_depths']}=={
        k:v for k,v in old.items() if k not in ['world','batch','rank_depths']}
    assert old==signature() and resize_depth_signature(new,world=32,batch=4)==old
    # Same total microbatch is insufficient: these worlds change the mixture.
    for world,batch in [(8,16),(64,2),(32,8)]:
        with pytest.raises(ValueError):resize_depth_signature(old,world=world,batch=batch)
    with pytest.raises(ValueError):resize_depth_signature(dict(old,rank_depths=[2]*32),world=16,batch=8)


def test_resized_ddp_has_same_masked_accumulated_global_gradient():
    torch.manual_seed(310)
    core=torch.nn.Linear(3,3,bias=False).double();x=torch.randn(256,3,dtype=torch.double)
    target=torch.randn_like(x);mask=(torch.arange(256)%5!=0).double();denominator=mask.sum()
    def ddp_gradient(s):
        worker_gradients=[]
        for rank,depth in enumerate(s['rank_depths']):
            model=copy.deepcopy(core);loss=torch.zeros((),dtype=torch.double)
            for micro in range(s['accumulation']):
                ids=routed_indices(0,micro,rank,s['world'],s['batch'],s['accumulation'],s['routing_seed'])
                state=torch.zeros_like(x[ids])
                for _ in range(depth):state=torch.tanh(model(state+x[ids]))
                loss+=((state-target[ids]).square().sum(-1)*mask[ids]).sum()
            (loss*s['world']/denominator).backward();worker_gradients.append(model.weight.grad)
        return torch.stack(worker_gradients).mean(0)
    old=signature();new=resize_depth_signature(old,world=16,batch=8)
    torch.testing.assert_close(ddp_gradient(old),ddp_gradient(new),rtol=1e-12,atol=1e-12)
