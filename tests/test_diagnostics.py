import copy
import torch
from compute_trainium.independent import IndependentModel
from compute_trainium.model import ModelConfig
from compute_trainium.diagnostics import traced_unroll


def test_instrumented_unroll_preserves_function_and_summed_core_gradients():
    torch.manual_seed(41)
    m=IndependentModel(ModelConfig(dim=16,layers=4,heads=2,kv_heads=1,ffn_dim=32,vocab_size=64,max_seq=16),repetitions=2,prelude=1,core=2,coda=1)
    x=torch.randint(0,64,(1,8));extra=torch.nn.ModuleList([copy.deepcopy(m.blocks[i]) for i in [1,2]])
    actual=m.hidden(x);traced,_,states=traced_unroll(m,x,2,extra)
    torch.testing.assert_close(actual,traced,rtol=0,atol=0)
    actual.square().sum().backward();shared=[p.grad.clone() for i in [1,2] for p in m.blocks[i].parameters()]
    m.zero_grad(set_to_none=True);traced.square().sum().backward()
    combined=[a.grad+b.grad for block,other in zip([m.blocks[1],m.blocks[2]],extra,strict=True) for a,b in zip(block.parameters(),other.parameters(),strict=True)]
    for a,b in zip(shared,combined,strict=True):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-4)
    assert len(states)==2
