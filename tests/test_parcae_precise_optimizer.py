import copy

import pytest
import torch

from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.parcae_optimizer_precise import ParcaePreciseOptimizer,independent_fp32_reference
from compute_trainium.parcae_update_qualification import configure_step,assign_gradient,compare_update,compare_moments,named_moments
from third_party.parcae_reference import muon_adamw as released


def test_explicit_fp32_recipe_matches_independent_modified_reference_and_recovery(monkeypatch):
    monkeypatch.setattr(released,'muon_step',independent_fp32_reference(released))
    monkeypatch.setattr(released,'adamw_step',getattr(released.adamw_step,'_torchdynamo_orig_callable',released.adamw_step))
    model=torch.nn.Sequential(torch.nn.Linear(8,12,bias=False),torch.nn.LayerNorm(12),torch.nn.Linear(12,8,bias=False))
    other=copy.deepcopy(model)
    expected=released.MuonAdamW(released.get_muon_param_groups(model.named_parameters(),adamw_lr=.008,adamw_betas=(.8,.95),adamw_eps=1e-10,adamw_wd=0.,muon_lr=.008))
    actual=ParcaePreciseOptimizer(other,gate_policy='release_muon')
    assert actual.recipe['specified_muon_polar_dtype']=='fp32'
    with pytest.raises(ValueError):ParcaeOptimizerCandidate(other,gate_policy='release_muon').load_state_dict(actual.state_dict())
    gradients={n:torch.randn(p.shape,generator=torch.Generator().manual_seed(739+i))*.03 for i,(n,p) in enumerate(model.named_parameters())}
    for step in range(3):
        before={n:p.detach().clone() for n,p in model.named_parameters()};old={n:p.detach().clone() for n,p in other.named_parameters()}
        for m,opt in [(model,expected),(other,actual)]:assign_gradient(m,gradients,configure_step(opt,step));opt.step()
        current={n:p.detach().clone() for n,p in model.named_parameters()};observed={n:p.detach().clone() for n,p in other.named_parameters()}
        assert compare_update(before,current,old,observed)['relative_l2']<2e-5
        assert compare_moments(named_moments(model,expected,reference=True),named_moments(other,actual))['relative_l2']<2e-6
    cloned=copy.deepcopy(other);restored=ParcaePreciseOptimizer(cloned,gate_policy='release_muon')
    restored.load_state_dict(copy.deepcopy(actual.state_dict()))
    for m,opt in [(other,actual),(cloned,restored)]:assign_gradient(m,gradients,1.);opt.step()
    assert all(torch.equal(a,b) for a,b in zip(other.parameters(),cloned.parameters(),strict=True))
