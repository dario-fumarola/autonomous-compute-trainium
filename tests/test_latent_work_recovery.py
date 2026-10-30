"""Launch recovery must not alter a scientific condition or extend its cap."""
import copy
import pytest
from scripts.recover_latent_exposure_work import recovered_plan,science,ID


def fixture():
    return dict(identity='old_unsafe',absolute_deadline_unix=1000,seconds_limit=43200,
        seed=71,world=64,batch=2,phase_seconds_limit=21600,cpu_phase_seconds_limit=1800,
        model_config={'layers':11},weight_decay=.1,data_report_sha256='data',
        arms={'ordinary':dict(phase='old_unsafe-fit',depth_schedule=[0]*12,lr=.001,
            real_targets=100,checkpoint_updates=[12],recovery_checkpoint_updates=[10],
            nominal_matrix_flops=1234,decay_start=10)})


def test_only_launch_metadata_changes_and_deadline_never_moves():
    old=fixture();p=recovered_plan(old,100,{}, {ID+'-fit':16000},{})
    assert science(old)==science(p)
    assert p['absolute_deadline_unix']==1000 and p['seconds_limit']==900
    assert set(p['identity']+'-fit')<=set('abcdefghijklmnopqrstuvwxyz0123456789-')
    assert p['arms']['ordinary']['phase']==ID+'-fit'
    for field,value in [('seed',72),('absolute_deadline_unix',1001),('weight_decay',.8)]:
        changed=copy.deepcopy(p);changed[field]=value;assert science(changed)!=science(old)
    changed=copy.deepcopy(p);changed['arms']['ordinary']['lr']=.003
    assert science(changed)!=science(old)


def test_expired_original_cap_cannot_be_reopened():
    with pytest.raises(TimeoutError):recovered_plan(fixture(),1000,{}, {},{})
