import numpy as np
import pytest

from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_scientific_schedule import update_work,matched_endpoint,source_endpoint,cooldown_start,schedule


def test_heterogeneous_work_counts_forward_tail_head_and_replicated_optimizer():
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=128,max_seq=16)
    n,d,f=8,c.dim,c.ffn_dim
    def block(i):return n*(8*d*d+4*d*f)+4*n*n*d+(2*n*32*c.heads if c.has_value_embedding(i) else 0)
    once=sum(block(i) for i in [0,1,4,5])+2*n*d*d+2*n*d*c.vocab_size
    repeated=sum(block(i) for i in [2,3])+2*n*d*d
    depths=np.array([[[[1,1],[2,1]],[[3,2],[2,2]]],[[[2,1],[1,1]],[[4,3],[1,1]]]],dtype=np.uint16)
    work=update_work(c,depths,recurrent=True,context=n,world=2,optimizer_per_replica=37)
    expected=[sum(3*once+(int(a)+2*int(b))*repeated for a,b in row.reshape(-1,2))+74 for row in depths]
    assert list(work)==expected
    ordinary=np.ones_like(depths)
    actual=update_work(c,ordinary,recurrent=False,context=n,world=2,optimizer_per_replica=37)
    assert list(actual)==[4*3*(sum(block(i) for i in range(6))+2*n*d*c.vocab_size)+74]*2
    with pytest.raises(ValueError):update_work(c,depths,recurrent=False,context=n,world=2,optimizer_per_replica=37)
    depths[0,0,0]=[1,2]
    with pytest.raises(ValueError):update_work(c,depths,recurrent=True,context=n,world=2,optimizer_per_replica=37)


def test_matched_work_source_and_decay_endpoints_are_explicit():
    result=matched_endpoint(np.array([10,11,12,13],dtype=np.int64),34)
    assert result['updates']==3 and result['executed_matmuls']==33
    assert result['next_update_would_exceed_budget']
    assert matched_endpoint(np.array([10,11,12]),33)['updates']==3
    with pytest.raises(ValueError):matched_endpoint(np.array([10,11]),30)
    with pytest.raises(ValueError):matched_endpoint(np.array([10,11]),9)
    with pytest.raises(OverflowError):matched_endpoint(np.array([2**62,2**62]),1)
    counts=np.arange(21,dtype=np.int64).reshape(7,3)
    assert source_endpoint(counts,3,2)=={'source_cursor':6,'domain_targets':[18,18,18],'real_targets':54}
    with pytest.raises(ValueError):source_endpoint(counts,4,2)
    assert cooldown_start(100)==50
    assert schedule(10,warmup_targets=20,cooldown_start_targets=50,end_targets=100,peak_lr=.008)['lr']==.004
    assert schedule(50,warmup_targets=20,cooldown_start_targets=50,end_targets=100,peak_lr=.008)['lr']==.008
    final=schedule(100,warmup_targets=20,cooldown_start_targets=50,end_targets=100,peak_lr=.008)
    assert final['lr']==0 and final['muon_cautious_decay']==0
