import torch
from torch.nn import functional as F

from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig,future_span_targets,future_span_loss_sum
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.training import PackedCorpus


def test_complete_window_adapter_preserves_ntp_and_two_independent_denominators():
    documents=[{'id':str(i),'domain':d,'split':'train','tokens':list(range(3,3+n))}
               for i,(d,n) in enumerate([('web',19),('code',31),('math',7)])]
    packed=PackedCorpus(documents,16)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    span=FutureSpanConfig(context=16,horizon=4,stride=3)
    torch.manual_seed(991);model=FutureSpanCandidate(config,span,fp16_body=False)
    objective=FutureSpanObjective(model,backend='eager')
    values=training_window(packed,0,config,span,2017)
    original=packed.batch([0],0)[0]
    for x,y in zip(values[:3],[original.inputs,original.targets,original.target_mask]):
        assert torch.equal(x,y)
    assert values[3].shape==(1,16,32)
    count=int(values[2].sum());anchors=int(values[-1].sum())
    coeff=objective_coefficients(count,anchors,.1,world=2)
    loss,parts=objective(*values,coeff,2,2)
    ntp,aux=model(values[0],values[3],total_steps=2,backward_cap=2)
    labels=future_span_targets(packed.tokens[:17].long()[None],packed.domains[:17][None]>0,span,64)
    ntp_loss=(F.cross_entropy(ntp.flatten(0,1),values[1].flatten(),reduction='none')*values[2].flatten()).sum()
    aux_loss=future_span_loss_sum(aux,labels,span)
    torch.testing.assert_close(parts,torch.stack([ntp_loss,aux_loss]))
    torch.testing.assert_close(loss,2*(ntp_loss/count+.1*aux_loss/anchors))
    loss.backward()
    assert model.future_lm_head.weight.grad.norm()>0
    assert model.transformer.wte.weight.grad.norm()>0


def test_empty_auxiliary_batch_and_runtime_weight_zero_are_finite():
    coeff=objective_coefficients(73,0,0.,world=32)
    assert coeff.dtype==torch.float32 and coeff[1]==0 and torch.isfinite(coeff).all()
