import copy

import pytest
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate, ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from scripts.qualify_native import digest


def model():
    return ParcaeCandidate(ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,
        vocab_size=64,max_seq=16),initialize=False)


def test_explicit_gate_policy_and_group_coverage():
    m = model()
    release = ParcaeOptimizerCandidate(m,gate_policy='release_muon')
    paper = ParcaeOptimizerCandidate(m,gate_policy='paper_adamw')
    names = {id(p):name for name,p in m.named_parameters()}
    release_gates = [names[id(p)] for g in release.param_groups if g['kind']=='muon'
                     for p in g['params'] if 've_gate' in names[id(p)]]
    paper_gates = [names[id(p)] for g in paper.param_groups if g['kind']=='muon'
                   for p in g['params'] if 've_gate' in names[id(p)]]
    assert len(release_gates) == 3 and not paper_gates
    actual = [id(p) for g in release.param_groups for p in g['params']]
    assert len(actual) == len(set(actual)) == len(list(m.parameters()))
    with pytest.raises(ValueError):
        paper.load_state_dict(release.state_dict())


def test_missing_gradient_stops_before_either_family_updates():
    m = model()
    opt = ParcaeOptimizerCandidate(m,gate_policy='release_muon')
    for p in m.parameters():
        p.grad = torch.ones_like(p)*.1
    list(m.parameters())[-1].grad = None
    before = digest({'model':m.state_dict(),'optimizer':opt.state_dict()})
    with pytest.raises(ValueError):
        opt.step()
    assert before == digest({'model':m.state_dict(),'optimizer':opt.state_dict()})


def test_complete_state_restore_preserves_next_update():
    torch.manual_seed(147)
    m = model()
    opt = ParcaeOptimizerCandidate(m,gate_policy='release_muon')
    for p in m.parameters():
        p.grad = torch.randn_like(p)*.03
    opt.step()
    other = copy.deepcopy(m)
    recovery = ParcaeOptimizerCandidate(other,gate_policy='release_muon')
    recovery.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert digest(opt.state_dict()) == digest(recovery.state_dict())
    for a,b in zip(m.parameters(),other.parameters(),strict=True):
        a.grad = torch.randn_like(a)*.1
        b.grad = a.grad.clone()
    opt.step()
    recovery.step()
    assert digest(m.state_dict()) == digest(other.state_dict())
    assert digest(opt.state_dict()) == digest(recovery.state_dict())
