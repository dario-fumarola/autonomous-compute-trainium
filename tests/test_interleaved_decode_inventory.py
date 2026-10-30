import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_sequential_reference import cached_sequential
from compute_trainium.tied_interleaved_ordinary import TiedOrdinary
from compute_trainium.interleaved_decode_inventory import decode_inventory


@pytest.mark.parametrize('heads,kv_heads',[(4,4),(4,2)])
def test_inventory_matches_executed_cached_reference_matrix_shapes(monkeypatch,heads,kv_heads):
    c=ModelConfig(dim=16,layers=2,heads=heads,kv_heads=kv_heads,ffn_dim=32,vocab_size=64,max_seq=8)
    m=InterleavedWorkspace(c).double();counts={'linear':0,'attention':0}
    linear,attention=F.linear,F.scaled_dot_product_attention
    def traced_linear(x,w,bias=None):
        counts['linear']+=2*(x.numel()//x.shape[-1])*w.shape[0]*w.shape[1]
        return linear(x,w,bias)
    def traced_attention(q,k,v,*args,**kwargs):
        counts['attention']+=4*q.shape[0]*q.shape[1]*q.shape[2]*k.shape[2]*q.shape[3]
        return attention(q,k,v,*args,**kwargs)
    monkeypatch.setattr(F,'linear',traced_linear);monkeypatch.setattr(F,'scaled_dot_product_attention',traced_attention)
    with torch.no_grad():cached_sequential(m,torch.tensor([[3,4,5,6],[7,8,9,10]]))
    inventories=[decode_inventory(c,t,latent=True,batch=2,bytes_per_element=8) for t in range(1,5)]
    assert counts['linear']==sum(v['body_projection_matmul_flops']+v['vocabulary_projection_matmul_flops'] for v in inventories)
    assert counts['attention']==sum(v['attention_matmul_flops'] for v in inventories)
    assert inventories[-1]['kv_bytes_at_context']==2*8*c.layers*2*(kv_heads*4)*8
    counts.update(linear=0,attention=0)
    with torch.no_grad():TiedOrdinary(c).double()(torch.tensor([[3],[4]]))
    ordinary=decode_inventory(c,1,latent=False,batch=2,bytes_per_element=8)
    assert sum(counts.values())==ordinary['forward_matmul_flops']


def test_unsupported_context_is_not_silently_extrapolated():
    c=ModelConfig(dim=16,layers=1,heads=2,kv_heads=2,ffn_dim=32,vocab_size=64,max_seq=8)
    for context in (0,9):
        with pytest.raises(ValueError):decode_inventory(c,context,latent=True)
