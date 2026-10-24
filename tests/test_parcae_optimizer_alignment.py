from collections import OrderedDict

import pytest
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_alignment import align_reference_groups
from compute_trainium.parcae_optimizer_candidate import parameter_groups
from compute_trainium.parcae_optimizer_precise import ParcaePreciseOptimizer
from compute_trainium.parcae_training_candidate import configure_optimizer
from compute_trainium.training import file_sha256
from scripts import audit_parcae_precise_training_point as auditor


def test_alignment_accepts_registration_order_but_rejects_changed_membership():
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=128,max_seq=8)
    model=ParcaeCandidate(config)
    groups=parameter_groups(model.named_parameters(),gate_policy='release_muon')
    lookup={id(p):name for name,p in model.named_parameters()}
    names=[[lookup[id(p)] for p in g['params']] for g in groups]
    model.transformer._modules=OrderedDict(reversed(list(model.transformer._modules.items())))
    other=parameter_groups(model.named_parameters(),gate_policy='release_muon')
    assert [[lookup[id(p)] for p in g['params']] for g in other]!=names
    aligned=align_reference_groups(model,other,names)
    assert [[lookup[id(p)] for p in g['params']] for g in aligned]==names
    corrupt=[list(g) for g in names];corrupt[0][0],corrupt[1][0]=corrupt[1][0],corrupt[0][0]
    with pytest.raises(ValueError,match='membership'):align_reference_groups(model,other,corrupt)
    corrupt=[list(g) for g in names];corrupt[0].append(corrupt[0][0])
    with pytest.raises(ValueError,match='duplicated'):align_reference_groups(model,other,corrupt)


@pytest.mark.parametrize('recurrent',[False,True])
def test_actual_update_audit_accepts_independently_ordered_model(tmp_path,monkeypatch,recurrent):
    torch.manual_seed(923)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=128,max_seq=8)
    model=ParcaeCandidate(config,recurrent=recurrent)
    preparation=tmp_path/'preparation';primary=tmp_path/'primary';output=tmp_path/'audit'
    for path in [preparation,primary,output]:path.mkdir()
    torch.save({'model':model.state_dict()},preparation/('initial-state.pt' if recurrent else 'initial-state-ordinary.pt'))
    raw={n:(torch.zeros_like(p) if '.c_q.' in n or '.fc.' in n else torch.randn_like(p)*.1) for n,p in model.named_parameters()}
    for n,p in model.named_parameters():p.grad=raw[n].clone()
    optimizer=ParcaePreciseOptimizer(model,gate_policy='release_muon',fused_adamw=True)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False))
    torch.save({'gradients':raw,'raw_gradient_norm':norm},primary/'initial-global-native.pt')
    configure_optimizer(optimizer,0);optimizer.step()
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},primary/'after-1-model-optimizer.pt')
    def independently_ordered(*_):
        reference=ParcaeCandidate(config,recurrent=recurrent)
        reference.transformer._modules=OrderedDict(reversed(list(reference.transformer._modules.items())))
        return reference
    monkeypatch.setattr(auditor,'make_reference',independently_ordered)
    for name in ['muon_step','adamw_step']:
        monkeypatch.setattr(auditor.reference_optimizer,name,getattr(auditor.reference_optimizer,name))
    native={'arm':'fixed1' if recurrent else 'ordinary','initial_global_gradient_sha256':file_sha256(primary/'initial-global-native.pt'),
            'first_checkpoint_sha256':file_sha256(primary/'after-1-model-optimizer.pt')}
    original=auditor.reference_optimizer.muon_step
    result=auditor.first_update(config,preparation,primary,native,output)
    assert result['passes_registered_gate']
    assert result['independent_group_membership_validated']
    assert result['reference_registration_order_aligned_by_name']
    assert auditor.reference_optimizer.muon_step is getattr(original,'_torchdynamo_orig_callable',original)
