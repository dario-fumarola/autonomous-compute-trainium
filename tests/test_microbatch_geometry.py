"""Numerical software fixtures,not generated research documents."""
import copy
import pytest
import torch

from compute_trainium.depth_mixture import routed_indices,window_permutation
from compute_trainium.microbatch_geometry import accumulation_for,stack_windows,physical_slot
from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.future_span_math_sdpa import FutureSpanMathSDPACandidate
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.training import PackedCorpus


def test_batch_geometries_cover_identical_global_source_order_and_logical_slots():
    for batch in [1,2,4]:
        accumulation=accumulation_for(batch);seen=[]
        for micro in range(accumulation):
            for rank in range(32):
                windows=routed_indices(1536,micro,rank,32,batch,accumulation,271828)
                for offset,window in enumerate(windows):
                    hw,sub=physical_slot(micro*batch+offset,batch)
                    assert routed_indices(1536,hw,rank,32,batch,accumulation,271828)[sub]==window
                seen.extend(windows)
        assert seen==[1536+x for x in window_permutation(1536,512,271828)]
    with pytest.raises(ValueError):accumulation_for(3)


@pytest.mark.parametrize('depth',[1,2])
def test_stacking_preserves_complete_objective_gradients_and_sample_isolation(depth):
    torch.manual_seed(783)
    packed=PackedCorpus([dict(id=str(i),domain=d,split='train',tokens=list(range(3,3+n)))
                        for i,(d,n) in enumerate([('web',13),('math',11),('code',17),('web',9)])],8)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=8)
    span=FutureSpanConfig(context=8,horizon=4,stride=2)
    model=FutureSpanMathSDPACandidate(config,span,fp16_body=False);other=copy.deepcopy(model)
    windows=[training_window(packed,i,config,span,2017) for i in range(4)]
    combined=stack_windows(windows)
    coefficient=objective_coefficients(int(combined[2].sum()),int(combined[-1].sum()),0.,1)
    loss,_=FutureSpanObjective(model)(*combined,coefficient,depth,depth)
    sequential=sum(FutureSpanObjective(other)(*w,coefficient,depth,depth)[0] for w in windows)
    torch.testing.assert_close(loss,sequential,rtol=2e-6,atol=2e-6)
    loss.backward();sequential.backward()
    for (_,p),(_,q) in zip(model.named_parameters(),other.named_parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6)
    with torch.no_grad():
        original,_=model(combined[0],combined[3],total_steps=depth,backward_cap=depth)
        changed=combined[0].clone();changed[0]=(changed[0]+1)%64
        altered,_=model(changed,combined[3],total_steps=depth,backward_cap=depth)
        torch.testing.assert_close(original[1:],altered[1:],rtol=0,atol=0)
