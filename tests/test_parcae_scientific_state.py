import copy

import pytest

from compute_trainium.parcae_scientific_state import validate_checkpoint,schedule_identity,configure_optimizer,worker_sources


def setup():
    case={'peak_lr':.008,'warmup_targets':20,'cooldown_start_targets':50,'end_targets':100,'total_updates':100}
    sig={'accumulation':16,'model':'fixture'}
    state={'purpose':'natural_low_depth_scientific_pilot','qualification_only':False,'signature':sig,
           'case_id':'main','schedule':schedule_identity(case),'stable_trunk':True,
           'update':40,'cursor':40*512,'real_targets':40,'executed_matmuls':80}
    return case,sig,state


def test_scientific_recovery_keeps_schedule_and_forbids_qualification_or_annealed_extension():
    case,sig,state=setup()
    validate_checkpoint(state,'main',case,sig,expected_targets=40,expected_matmuls=80)
    branch={**case,'cooldown_start_targets':40,'end_targets':80,'total_updates':80,'parent':{'case_id':'main','update':40}}
    validate_checkpoint(state,'short',branch,sig,expected_targets=40,expected_matmuls=80)
    state['stable_trunk']=False
    with pytest.raises(ValueError,match='annealed'):validate_checkpoint(state,'short',branch,sig,expected_targets=40,expected_matmuls=80)
    # Exact same-run recovery inside its own cooldown is still valid.
    validate_checkpoint(state,'main',case,sig,expected_targets=40,expected_matmuls=80)
    state['qualification_only']=True
    with pytest.raises(ValueError,match='Qualification'):validate_checkpoint(state,'main',case,sig,expected_targets=40,expected_matmuls=80)
    state['qualification_only']=False;changed={**case,'peak_lr':.016}
    with pytest.raises(ValueError,match='schedule'):validate_checkpoint(state,'main',changed,sig,expected_targets=40,expected_matmuls=80)
    with pytest.raises(ValueError,match='counter'):validate_checkpoint(state,'main',case,sig,expected_targets=41,expected_matmuls=80)


def test_scientific_optimizer_schedule_preserves_both_families_and_zero_endpoint():
    case,_,_=setup()
    class Optimizer:
        param_groups=[{'kind':'adamw','base_lr':1.},{'kind':'muon','base_lr':1.}]
    opt=Optimizer();configure_optimizer(opt,case,10)
    assert opt.param_groups[0]['lr']==.004 and opt.param_groups[1]['lr']==.004
    assert opt.param_groups[1]['weight_decay']==.2
    configure_optimizer(opt,case,100)
    assert all(g['lr']==0 for g in opt.param_groups)
    assert opt.param_groups[1]['weight_decay']==0


def test_qualification_driver_states_remain_ineligible_for_science():
    case,sig,state=setup();sig={**sig,'execution_purpose':'low_depth_driver_integration_qualification'}
    state.update(purpose=sig['execution_purpose'],qualification_only=True,signature=sig)
    validate_checkpoint(state,'main',case,sig,expected_targets=40,expected_matmuls=80)
    science={**sig,'execution_purpose':'natural_low_depth_scientific_pilot'}
    with pytest.raises(ValueError,match='Qualification'):
        validate_checkpoint(state,'main',case,science,expected_targets=40,expected_matmuls=80)


def test_worker_validates_frozen_code_without_requiring_uncopied_docs_or_tests():
    source={'scripts/train_low_depth_scientific.py':'a','compute_trainium/model.py':'b',
            'docs/protocol.md':'c','tests/test_model.py':'d'}
    runtime={k:v for k,v in source.items() if k.startswith(('scripts/','compute_trainium/'))}
    assert worker_sources({'source_sha256':source,'worker_source_sha256':runtime})==runtime
    with pytest.raises(ValueError):worker_sources({'source_sha256':source,'worker_source_sha256':source})
