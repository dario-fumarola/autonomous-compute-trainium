import torch
from torch.nn import functional as F

from compute_trainium.future_span import (
    FutureSpanConfig,FutureSpanTargets,FutureSpanCandidate,
    future_span_targets,future_span_loss_sum,normalized_objective,
)
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_fp16_candidate import ParcaeFP16Candidate
from compute_trainium.parcae_optimizer_candidate import parameter_groups


def test_presence_targets_exclude_next_token_and_reject_document_boundaries():
    config=FutureSpanConfig(context=8,horizon=4,stride=4)
    source=torch.tensor([[0,1,2,2,3,4,5,6,7]])
    real=torch.ones_like(source,dtype=torch.bool);real[:,5]=False
    labels=future_span_targets(source,real,config,16)
    assert labels.positions.tolist()==[0,4]
    assert labels.token_ids.tolist()==[[[2,2,3],[5,6,7]]]
    assert labels.valid.tolist()==[[True,False]]
    changed=source.clone();changed[:,1]=15
    assert torch.equal(labels.token_ids[:,0],future_span_targets(changed,real,config,16).token_ids[:,0])
    logits=torch.randn(1,2,16,requires_grad=True)
    expected=torch.zeros(16);expected[[2,3]]=1
    torch.testing.assert_close(future_span_loss_sum(logits,labels,config),
        F.binary_cross_entropy_with_logits(logits[0,0],expected,reduction='sum')/3)
    empty=future_span_targets(source,torch.zeros_like(real),config,16)
    loss=future_span_loss_sum(logits,empty,config);assert loss.item()==0
    loss.backward();assert logits.grad.count_nonzero()==0


def test_global_denominators_match_uneven_microbatches_and_ddp_mean():
    config=FutureSpanConfig(context=8,horizon=4,stride=4)
    source=torch.arange(9).repeat(4,1);real=torch.ones_like(source,dtype=torch.bool)
    real[1,:]=False;real[2,5]=False
    targets=future_span_targets(source,real,config,16)
    logits=torch.randn(4,2,16,requires_grad=True)
    ntp=torch.randn(4,requires_grad=True)
    count=int(targets.valid.sum());full=normalized_objective(ntp.sum(),future_span_loss_sum(logits,targets,config),
        ntp_count=19,future_count=count,weight=.1)
    local=[]
    for rank in range(2):
        micro=[]
        for row in [rank,rank+2]:
            sub=FutureSpanTargets(targets.positions,targets.token_ids[row:row+1],targets.valid[row:row+1])
            micro.append(normalized_objective(ntp[row],future_span_loss_sum(logits[row:row+1],sub,config),
                ntp_count=19,future_count=count,weight=.1,world=2))
        local.append(sum(micro))
    torch.testing.assert_close(full,sum(local)/2)
    expected=torch.autograd.grad(full,(logits,ntp),retain_graph=True)
    observed=torch.autograd.grad(sum(local)/2,(logits,ntp))
    for a,b in zip(expected,observed):torch.testing.assert_close(a,b)


def model_pair(context=8):
    torch.manual_seed(17)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=context)
    future=FutureSpanConfig(context=context,horizon=4,stride=2)
    candidate=FutureSpanCandidate(config,future,fp16_body=False)
    base=ParcaeFP16Candidate(config,fp16_body=False,initialize=False)
    base.load_state_dict(candidate.prediction_state_dict(),strict=True)
    return config,future,candidate,base


def test_zero_weight_control_has_exact_ntp_gradients_and_removable_head():
    config,future,candidate,base=model_pair()
    source=torch.arange(9)[None];state=torch.randn(1,8,32)
    labels=future_span_targets(source,torch.ones_like(source,dtype=torch.bool),future,config.vocab_size)
    logits,aux=candidate(source[:,:-1],state,total_steps=2,backward_cap=2)
    original=base(source[:,:-1],state,total_steps=2,backward_cap=2)
    torch.testing.assert_close(logits,original,rtol=0,atol=0)
    loss=F.cross_entropy(logits.flatten(0,1),source[:,1:].flatten(),reduction='sum')
    combined=normalized_objective(loss,future_span_loss_sum(aux,labels,future),
        ntp_count=8,future_count=int(labels.valid.sum()),weight=0)
    combined.backward();F.cross_entropy(original.flatten(0,1),source[:,1:].flatten()).backward()
    for name,p in base.named_parameters():
        torch.testing.assert_close(dict(candidate.named_parameters())[name].grad,p.grad,rtol=0,atol=0)
    assert candidate.future_lm_head.weight.grad.count_nonzero()==0
    assert candidate.future_lm_head.bias.grad.count_nonzero()==0
    groups=parameter_groups(candidate.named_parameters(),gate_policy='release_muon')
    assert any(p is candidate.future_lm_head.weight for p in groups[0]['params'])
    assert any(p is candidate.future_lm_head.bias for p in groups[0]['params'])
    base.zero_grad();candidate.zero_grad()
    _,aux=candidate(source[:,:-1],state,total_steps=2,backward_cap=2)
    future_span_loss_sum(aux,labels,future).backward()
    assert candidate.future_lm_head.weight.grad.norm()>0
    assert candidate.transformer.wte.weight.grad.norm()>0


def test_future_labels_change_without_changing_prefix_predictions_at_supported_depths():
    config,future,candidate,_=model_pair(16)
    source=torch.arange(17)[None];changed=source.clone();changed[:,9:]+=17
    real=torch.ones_like(source,dtype=torch.bool)
    original_labels=future_span_targets(source,real,future,config.vocab_size)
    changed_labels=future_span_targets(changed,real,future,config.vocab_size)
    assert not torch.equal(original_labels.token_ids[:,4],changed_labels.token_ids[:,4])
    state=torch.randn(1,16,32)
    for depth in [1,2,3]:
        a,b=candidate(source[:,:-1],state,total_steps=depth,backward_cap=depth)
        c,d=candidate(changed[:,:-1],state,total_steps=depth,backward_cap=depth)
        torch.testing.assert_close(a[:,:9],c[:,:9],rtol=0,atol=0)
        torch.testing.assert_close(b[:,:5],d[:,:5],rtol=0,atol=0)
