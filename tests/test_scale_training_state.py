import copy

import pytest

from compute_trainium.scale_training_state import PURPOSE,validate_saved,validate_lineage
from compute_trainium.scale_training_schedule import rates


def test_training_state_rejects_quarantine_cross_seed_data_and_counter_changes():
    signature=dict(study_sha256='frozen',case='r',phase='main',initializer_sha256='fresh-c0',
        packing_fingerprint='accepted',model_role='fixed_two',state_seed=123)
    value=dict(purpose=PURPOSE,signature=signature,eligible_for_science=True,qualification_only=False,
        rank=1,world=64,update=20000,cursor=10240000,work=20000*10**15,optimizer={'lineage':signature,'execution':{'update':20000}})
    validate_saved(value,signature,1,64,20000,10**15)
    for key,change in [('purpose','scale_large_model_qualification_only'),('eligible_for_science',False),
                       ('rank',2),('cursor',10239999),('work',20000*10**15-1),('qualification_only',True)]:
        with pytest.raises(ValueError):validate_saved({**value,key:change},signature,1,64,20000,10**15)
    for key in ['phase','initializer_sha256','packing_fingerprint','state_seed']:
        changed=copy.deepcopy(signature);changed[key]='different'
        with pytest.raises(ValueError):validate_saved(value,changed,1,64,20000,10**15)
    with pytest.raises(ValueError):validate_lineage({'phase':'main'})


def test_shared_stable_trunks_and_independent_cooldowns_end_at_zero():
    common=dict(warmup_targets=100,peak_lr=.032)
    a=rates(400,**common,cooldown_targets=450,end_targets=500)
    b=rates(400,**common,cooldown_targets=630,end_targets=700)
    assert a==b
    end=rates(500,**common,cooldown_targets=450,end_targets=500)
    assert end['lr']==0 and end['weight_decay']==0
    assert rates(25,**common,cooldown_targets=450,end_targets=500)['momentum']==pytest.approx(.875)


def test_rotating_backup_retires_only_after_verified_successor(tmp_path):
    import json
    from compute_trainium.training import file_sha256
    from compute_trainium.scale_training_state import retire_verified_periodic
    signature=dict(study_sha256='frozen',case='r',phase='main',initializer_sha256='fresh',
        packing_fingerprint='accepted',model_role='fixed_two',state_seed=12)
    def checkpoint(update):
        file=tmp_path/f'rank0-u{update}.pt';file.write_bytes(bytes([update//1000])*64)
        note=dict(rank=0,file=file.name,sha256=file_sha256(file))
        manifest=dict(purpose=PURPOSE,signature=signature,update=update,world=1,
            all_replicated_states_equal=True,shards=[note])
        (tmp_path/f'manifest-u{update}.json').write_text(json.dumps(manifest));return file
    old=checkpoint(1000);current=checkpoint(2000)
    current.write_bytes(b'corrupt')
    with pytest.raises(AssertionError):retire_verified_periodic(tmp_path,1000,2000,signature,[])
    assert old.exists()
    current=checkpoint(2000)
    with pytest.raises(ValueError):retire_verified_periodic(tmp_path,1000,2000,signature,[1000])
    assert old.exists()
    retire_verified_periodic(tmp_path,1000,2000,signature,[])
    assert not old.exists() and current.exists() and (tmp_path/'manifest-u1000.json').exists()
    report=json.loads((tmp_path/'periodic-retirement-u1000.json').read_text())
    assert not report['permanent_states_released'] and len(report['original_shards'])==1
