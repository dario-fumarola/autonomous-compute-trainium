import copy

import pytest
import torch

from compute_trainium.parcae_candidate import ParcaeCandidate,ParcaeCandidateConfig
from compute_trainium.parcae_optimizer_precise import ParcaePreciseOptimizer
from compute_trainium.parcae_training_candidate import configure_optimizer
from compute_trainium.training import file_sha256
from scripts import audit_parcae_precise_training_point as auditor
from scripts.capture_parcae_optimizer_precise import eager_fp32_optimizer
from scripts.run_parcae_fp16_precise_native import launch_ready
from scripts.run_parcae_precise_training_qualification import launch_policy
from scripts.run_parcae_optimizer_precise_native import launch_ready as optimizer_ready


def test_new_queues_require_explicit_fp32_optimizer_proof_and_keep_deadlines():
    g={'status':'PASS_PAIRED_NATIVE_PARCAE_STATIC_STAGE_GLOBAL_GRADIENTS'}
    old={'status':'PASS_PAIRED_NATIVE_PARCAE_OPTIMIZER_AND_RECOVERY'}
    new={'status':'PASS_PAIRED_NATIVE_PARCAE_FP32_POLAR_OPTIMIZER_AND_RECOVERY'}
    f={'status':'PASS_PAIRED_NATIVE_PARCAE_FP16_BODY_GRADIENTS'}
    for call in [lambda o:launch_ready(g,o,False,1,2),lambda o:launch_policy(g,o,f,False,1,2)]:
        with pytest.raises(RuntimeError):call(old)
    assert launch_ready(g,new,False,1,2)
    assert not launch_ready(g,new,True,1,2)
    assert launch_policy(g,new,f,False,1,2)==['fp32','fp16']
    with pytest.raises(TimeoutError):launch_ready(g,new,False,3,2)
    model=torch.nn.Linear(8,12,bias=False)
    optimizer=eager_fp32_optimizer(model,gate_policy='release_muon',fused_adamw=True,muon_backend='neuron')
    assert optimizer.recipe['specified_muon_polar_dtype']=='fp32'
    assert optimizer.muon.tensor_step.__name__=='muon_tensor_step_fp32'
    assert not optimizer_ready(None,False,1,2)
    endpoints={'status':'COMPLETE_NATIVE_SCIENTIFIC_ENDPOINTS'}
    assert optimizer_ready(endpoints,False,1,2)
    assert not optimizer_ready(endpoints,True,1,2)
    with pytest.raises(RuntimeError):optimizer_ready(g,False,1,2)


@pytest.mark.parametrize('recurrent',[False,True])
def test_independent_first_actual_update_uses_fp32_polar_in_both_arms(tmp_path,monkeypatch,recurrent):
    # Numerical software fixture only. No example enters training or evaluation.
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
    monkeypatch.setattr(auditor,'make_reference',lambda *_:ParcaeCandidate(config,recurrent=recurrent))
    for name in ['muon_step','adamw_step']:
        monkeypatch.setattr(auditor.reference_optimizer,name,getattr(auditor.reference_optimizer,name))
    native={'arm':'fixed1' if recurrent else 'ordinary','initial_global_gradient_sha256':file_sha256(primary/'initial-global-native.pt'),
            'first_checkpoint_sha256':file_sha256(primary/'after-1-model-optimizer.pt')}
    result=auditor.first_update(config,preparation,primary,native,output)
    assert result['passes_registered_gate']
    assert result['prescribed_cpu_optimizer_steps']==1
    assert result['research_optimizer_updates']==0
