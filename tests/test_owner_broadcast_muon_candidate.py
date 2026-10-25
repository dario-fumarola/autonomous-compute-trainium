"""Real two-process CPU collective tests on numerical software fixtures."""
import copy
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from compute_trainium.owner_broadcast_muon_candidate import OwnerBroadcastOptimizerCandidate
from compute_trainium.parcae_runtime_scalars import ParcaeRuntimeScalarOptimizer


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte=nn.Embedding(11,4)
        self.square=nn.Linear(4,4,bias=False)
        self.square_second=nn.Linear(4,4,bias=False)
        self.tall=nn.Linear(4,7,bias=False)
        self.wide=nn.Linear(7,4,bias=False)
        self.ln_f=nn.LayerNorm(4)


def compare(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:compare(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):compare(x,y)
    else:assert a==b


def worker(rank,store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(914);reference=Fixture();candidate=copy.deepcopy(reference)
        a=ParcaeRuntimeScalarOptimizer(reference,gate_policy='release_muon')
        b=OwnerBroadcastOptimizerCandidate(candidate,gate_policy='release_muon')
        for step in range(5):
            generator=torch.Generator().manual_seed(55+step)
            for index,(p,q) in enumerate(zip(reference.parameters(),candidate.parameters(),strict=True)):
                gradient=torch.randn(p.shape,generator=generator)
                if step==3 or (step==0 and index%2==0):gradient.zero_()
                p.grad=gradient.clone();q.grad=gradient.clone()
            for optimizer in [a,b]:
                for group in optimizer.param_groups:
                    group['lr']=.001*(step+1)
                    if group['kind']=='muon':
                        group['momentum']=.85+.02*step;group['weight_decay']=.2-.03*step
            a.step();b.step()
            compare(reference.state_dict(),candidate.state_dict())
            compare(a.state_dict(),b.state_dict())
            assert b.muon.last_owned_group_indices==[i for i in range(len(b.muon.param_groups)) if i%2==rank]
            if step==2:
                restored=OwnerBroadcastOptimizerCandidate(candidate,gate_policy='release_muon')
                restored.load_state_dict(copy.deepcopy(b.state_dict()));b=restored
        # Both ranks also retain identical complete optimizer history.
        for value in list(candidate.parameters())+[v for state in b.muon.state.values() for v in state.values()]:
            gold=value.detach().clone();dist.broadcast(gold,src=0)
            torch.testing.assert_close(value,gold,rtol=0,atol=0)
    finally:dist.destroy_process_group()


def test_two_rank_owner_updates_and_all_moments_equal_unsharded_recipe(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'gloo-store'),),nprocs=2,join=True)


def test_missing_process_group_fails_before_any_state_or_parameter_mutation():
    model=Fixture();before=copy.deepcopy(model.state_dict())
    optimizer=OwnerBroadcastOptimizerCandidate(model,gate_policy='release_muon')
    for p in model.parameters():p.grad=torch.ones_like(p)
    try:optimizer.step()
    except RuntimeError as error:assert 'process group' in str(error)
    else:raise AssertionError('Distributed initialization is required')
    compare(before,model.state_dict())
    assert not optimizer.adamw.state and not optimizer.muon.state
