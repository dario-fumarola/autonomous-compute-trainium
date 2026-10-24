import copy

import pytest
import torch

from compute_trainium.parcae_recovery_check import check_replay,moments
from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_candidate import ParcaeOptimizerCandidate
from compute_trainium.parcae_training_candidate import configure_optimizer
from compute_trainium.training import file_sha256
from scripts import audit_parcae_training_point as auditor


def fixture():
    optimizer={'recipe':{'groups':[['transformer.wte.weight'],['core.weight']]},
        'adamw':{'param_groups':[{'params':[0]}],'state':{0:{'step':torch.tensor(120.),
            'exp_avg':torch.ones(2,2),'exp_avg_sq':torch.ones(2,2)*.1}}},
        'muon':{'param_groups':[{'params':[0]}],'state':{0:{'m':torch.ones(1,2,2),'v':torch.ones(1,2,1)*.1}}}}
    return {'model':{'transformer.wte.weight':torch.ones(2,2)*100.,'lm_head.weight':torch.ones(2,2)*100.,
        'core.weight':torch.ones(2,2),'freqs_cis':torch.zeros(4)},'optimizer':optimizer}


def test_replay_preserves_aliases_buffers_and_both_optimizer_families():
    before=fixture();end=copy.deepcopy(before)
    for name in ['transformer.wte.weight','lm_head.weight']:end['model'][name]-=.1
    end['model']['core.weight']-=.001
    replay=copy.deepcopy(end)
    row=check_replay(before,end,replay)
    assert row['passes_two_percent_gate']
    assert set(moments(end['optimizer']))=={'transformer.wte.weight','core.weight'}
    replay['model']['core.weight']+=.0002
    row=check_replay(before,end,replay)
    assert row['update']['relative_l2']<.002
    assert not row['passes_two_percent_gate']
    assert row['families']['muon']['update']['relative_l2']>.19
    replay=copy.deepcopy(end);replay['model']['freqs_cis'][0]=1
    with pytest.raises(ValueError,match='buffer'):check_replay(before,end,replay)
    replay=copy.deepcopy(end);replay['model']['lm_head.weight'][0,0]+=1
    with pytest.raises(ValueError,match='Tied'):check_replay(before,end,replay)


@pytest.mark.parametrize('recurrent',[False,True])
def test_first_update_auditor_checks_real_optimizer_step_and_zero_matrices(tmp_path,monkeypatch,recurrent):
    torch.manual_seed(923)
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=128,max_seq=8)
    model=ParcaeCandidate(config,recurrent=recurrent)
    preparation=tmp_path/'preparation';primary=tmp_path/'primary';output=tmp_path/'audit'
    for path in [preparation,primary,output]:path.mkdir()
    torch.save({'model':model.state_dict()},preparation/('initial-state.pt' if recurrent else 'initial-state-ordinary.pt'))
    raw={n:(torch.zeros_like(p) if '.c_q.' in n or '.fc.' in n else torch.randn_like(p)*.1)
         for n,p in model.named_parameters()}
    for n,p in model.named_parameters():p.grad=raw[n].clone()
    optimizer=ParcaeOptimizerCandidate(model,gate_policy='release_muon',fused_adamw=True)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False))
    torch.save({'gradients':raw,'raw_gradient_norm':norm},primary/'initial-global-native.pt')
    configure_optimizer(optimizer,0);optimizer.step()
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},primary/'after-1-model-optimizer.pt')
    # Constructor substitution keeps the unit independent of a private clone;
    # the arithmetic reference remains the separately vendored released optimizer.
    monkeypatch.setattr(auditor,'make_reference',lambda *_:ParcaeCandidate(config,recurrent=recurrent))
    native={'arm':'fixed1' if recurrent else 'ordinary','initial_global_gradient_sha256':file_sha256(primary/'initial-global-native.pt'),
            'first_checkpoint_sha256':file_sha256(primary/'after-1-model-optimizer.pt')}
    result=auditor.first_update(config,preparation,primary,native,output)
    assert result['passes_registered_gate']
    assert result['prescribed_cpu_optimizer_steps']==1
    assert result['research_optimizer_updates']==0
