import json
import pytest
import torch
from compute_trainium.bounded_replica_capture import capture_groups,capture_replicas
from compute_trainium.scale_sharded_fixture import cpu_tree
from scripts.qualify_native import digest


def test_capture_groups_cover_each_rank_once_with_bounded_host_copies():
    groups=capture_groups(64,4)
    assert [r for group in groups for r in group]==list(range(64))
    assert max(map(len,groups))==4
    with pytest.raises(ValueError):capture_groups(64,64)


def test_capture_preserves_exact_full_state_and_next_optimizer_step(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
    torch.manual_seed(73);a=torch.nn.Linear(4,3);b=torch.nn.Linear(4,3);b.load_state_dict(a.state_dict())
    oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
    x=torch.randn(7,4)
    for m,o in ((a,oa),(b,ob)):m(x).square().mean().backward();o.step()
    expected=cpu_tree(dict(model=a.state_dict(),optimizer=oa.state_dict(),rng=torch.get_rng_state()))
    saved=[]
    h=capture_replicas(a,oa,rank=0,world=1,rank_batch=1,output=tmp_path,update=1,fallback_ops=[],
        on_primary=lambda state,sha:saved.append((state,sha)))
    assert h==digest(expected)==saved[0][1]
    assert json.loads((tmp_path/'replica-0.json').read_text())['digest']==h
    for m,o in ((a,oa),(b,ob)):o.zero_grad(set_to_none=True);m(x).square().mean().backward();o.step()
    assert digest(cpu_tree(dict(model=a.state_dict(),optimizer=oa.state_dict(),rng=torch.get_rng_state())))==digest(cpu_tree(dict(model=b.state_dict(),optimizer=ob.state_dict(),rng=torch.get_rng_state())))
