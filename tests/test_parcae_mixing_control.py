"""Numerical software fixtures only; no generated research examples."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import MixingOnlyFutureSpanCandidate,mixing_only_ledger


def models(fp16=False):
    torch.manual_seed(197)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,
        prelude=1,core=2,coda=1,max_seq=16)
    future=FutureSpanConfig(context=16,horizon=4,stride=2)
    old=FutureSpanMathSDPACandidate(config,future,initialize=False,fp16_body=fp16)
    control=MixingOnlyFutureSpanCandidate(config,future,initialize=False,fp16_body=fp16)
    control.load_state_dict(old.state_dict(),strict=True)
    return config,old,control,torch.arange(16)[None]%41,torch.randn(1,16,32)


@pytest.mark.parametrize('fp16',[False,True])
def test_one_step_preserves_exact_function_inventory_and_complete_gradient(fp16):
    _,old,control,tokens,noise=models(fp16)
    assert list(old.state_dict()) == list(control.state_dict())
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in control.parameters())
    a=old(tokens,noise,total_steps=1,backward_cap=1)
    b=control(tokens,noise,total_steps=1,backward_cap=1)
    for x,y in zip(a,b,strict=True):torch.testing.assert_close(x,y,rtol=0,atol=0)
    sum(x.square().mean() for x in a).backward()
    sum(x.square().mean() for x in b).backward()
    for p,q in zip(old.parameters(),control.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_multiple_mixes_execute_core_once_keep_all_gradient_paths_and_are_causal():
    _,old,control,tokens,noise=models()
    counts={'mix':0,'core':0}
    handles=[control.transformer.adapter.register_forward_hook(lambda *args:counts.__setitem__('mix',counts['mix']+1))]
    handles += [b.register_forward_hook(lambda *args:counts.__setitem__('core',counts['core']+1))
                for b in control.transformer.core_block]
    state=noise.clone().requires_grad_()
    out=control(tokens,state,total_steps=3,backward_cap=3)
    assert counts == {'mix':3,'core':2}
    for h in handles:h.remove()
    # Independent explicit expression using the original model's modules.
    e=old.encode(tokens)
    h=noise.clone().requires_grad_()
    first=old.transformer.adapter(h,e);first.retain_grad()
    h2=old.transformer.adapter(old.transformer.adapter(first,e),e)
    for j,block in enumerate(old.transformer.core_block):
        h2=old._apply_block(block,h2,tokens,old.config.prelude+j)
    reference=old.decode(h2,tokens)
    for x,y in zip(out,reference,strict=True):torch.testing.assert_close(x,y,rtol=0,atol=0)
    sum(x.square().mean() for x in out).backward()
    sum(x.square().mean() for x in reference).backward()
    assert first.grad.norm() > 0 and state.grad.norm() > 0
    for p,q in zip(old.parameters(),control.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    changed=tokens.clone();changed[:,9:]=(changed[:,9:]+7)%41
    with torch.no_grad():
        modified=control(changed,noise,total_steps=3,backward_cap=3)[0]
    torch.testing.assert_close(out[0][:,:9],modified[:,:9],rtol=0,atol=0)
    with pytest.raises(ValueError):control(tokens,noise,total_steps=3,backward_cap=1)


def test_forward_matrix_ledger_matches_independent_operator_trace():
    config,_,control,tokens,noise=models()
    class Counter(TorchDispatchMode):
        def __init__(self):self.work=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func in [torch.ops.aten.mm.default,torch.ops.aten.bmm.default]:
                a,b=args[:2]
                self.work+=2*a.numel()*b.shape[-1]
            elif func == torch.ops.aten.addmm.default:
                a,b=args[1:3]
                self.work+=2*a.numel()*b.shape[-1]
            return func(*args,**(kwargs or {}))
    for depth in [1,2,3,5]:
        counter=Counter()
        # no_grad keeps ATen linear/matmul decompositions visible to dispatch;
        # inference_mode presents opaque composite operators instead.
        with torch.no_grad(),counter:
            control(tokens,noise,total_steps=depth,backward_cap=depth)
        auxiliary=2*len(control.future_config.positions)*config.dim*config.vocab_size
        expected=mixing_only_ledger(config,16,total_steps=depth,backward_cap=depth)
        assert counter.work == expected['forward_matmul_flops']+auxiliary
