import copy
import pytest
import torch
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.interleaved_sequential_reference import cached_sequential
from compute_trainium.interleaved_last_readout import cached_last_logits


@pytest.mark.parametrize('length',[1,5])
def test_final_only_head_preserves_full_prefix_logits_and_all_gradients(length):
    torch.manual_seed(10501);c=ModelConfig(dim=16,layers=2,heads=2,kv_heads=1,ffn_dim=32,vocab_size=23,max_seq=8)
    a=InterleavedWorkspace(c).double();b=copy.deepcopy(a);x=torch.randint(23,(2,length));y=torch.randint(23,(2,))
    out=cached_last_logits(a,x);reference=cached_sequential(b,x)[:,-1]
    torch.testing.assert_close(out,reference,atol=1e-10,rtol=1e-10)
    torch.nn.functional.cross_entropy(out,y).backward();torch.nn.functional.cross_entropy(reference,y).backward()
    for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n==m;torch.testing.assert_close(p.grad,q.grad,atol=1e-9,rtol=1e-8)
