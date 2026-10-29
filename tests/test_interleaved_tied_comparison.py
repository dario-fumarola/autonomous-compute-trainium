import copy
import pytest
from scripts.compare_interleaved_tied import compare,inference
from compute_trainium.model import ModelConfig


def fixtures():
    # Software-only scalar fixtures, never research examples.
    rows=[dict(id=str(i),index=i,group='group'+str(i),domain=('web','math','code')[i//128],targets=512,loss_sum=512*3.) for i in range(384)]
    control=copy.deepcopy(rows)
    for r in control:r['loss_sum']+=512*{'web':1.,'math':2.,'code':-1.}[r['domain']]
    return rows,control


def test_pairing_keeps_registered_domain_weights_and_direction():
    candidate,control=fixtures();r=compare(candidate,control,replicates=10)
    assert r['comparison']['gain']==pytest.approx(.8)
    assert r['comparison']['conditional_95_interval']==pytest.approx([.8,.8])
    assert r['domains']['code']['gain']==pytest.approx(-1.)


@pytest.mark.parametrize('defect',['duplicate','domain','group','targets','nan'])
def test_pairing_rejects_incomplete_or_changed_examples(defect):
    candidate,control=fixtures()
    if defect=='duplicate':control[0]=control[1].copy()
    elif defect=='domain':control[0]['domain']='math'
    elif defect=='group':control[0]['group']='different'
    elif defect=='targets':control[0]['targets']=511
    else:control[0]['loss_sum']=float('nan')
    with pytest.raises(AssertionError):compare(candidate,control,replicates=10)


def test_cached_ledger_counts_causal_attention_and_shared_head():
    c=ModelConfig(dim=512,layers=11,heads=8,kv_heads=8,ffn_dim=1376,vocab_size=32000,max_seq=512)
    r=inference(c,False)
    assert r['nominal_cached_forward_matmul_flops']==55353802752
    assert r['fp32_kv_bytes']==23068672
    body_without_attention=c.flops(0,512)['forward_matmul_flops']-4*c.layers*c.dim*512**2
    direct=body_without_attention+sum(4*c.layers*c.dim*(i+1) for i in range(512))+2*512*c.dim*c.vocab_size
    assert r['nominal_cached_forward_matmul_flops']==direct
