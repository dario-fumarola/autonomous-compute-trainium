import pytest
import torch

from compute_trainium.checkpoint_migration import migration_signature,optimizer_implementation


def test_execution_resize_preserves_global_microbatches_and_all_other_settings():
    old=dict(world=64,batch=2,accumulation=2,context=1024,seed=23,
             corpus_sha256='old',packing_sha256='old-pack',optimizer='plain')
    new=migration_signature(old,world=32,batch=4,accumulation=2,corpus_sha='new',packing_sha='new-pack',optimizer='fused')
    assert new['world']*new['batch']==old['world']*old['batch']==128
    assert new['context']==old['context'] and new['seed']==23
    with pytest.raises(ValueError):
        migration_signature(old,world=32,batch=2,accumulation=4,corpus_sha='new',packing_sha='new-pack',optimizer='fused')


def test_optimizer_metadata_conversion_preserves_moments_counters_and_hyperparameters():
    original={'state':{0:{'step':torch.tensor(1152.),'exp_avg':torch.randn(9),'exp_avg_sq':torch.rand(9)}},
              'param_groups':[{'params':[0],'lr':.001,'betas':(.9,.999),'eps':1e-8,'weight_decay':.01,'foreach':False,'fused':False}]}
    new=optimizer_implementation(original,'fused')
    assert new['param_groups'][0]['fused'] and not original['param_groups'][0]['fused']
    for key,value in original['state'][0].items():assert torch.equal(new['state'][0][key],value)
    for key in ['params','lr','betas','eps','weight_decay']:assert new['param_groups'][0][key]==original['param_groups'][0][key]


def test_full_migration_preserves_history_and_rejects_annealed_parent():
    from dataclasses import asdict
    from types import SimpleNamespace
    from compute_trainium.continuation import prefix_fingerprint
    from compute_trainium.training import WSDSchedule
    from scripts.migrate_checkpoint import migrate
    from scripts.qualify_native import digest
    tokens=torch.arange(17,dtype=torch.int32);domains=torch.ones(17,dtype=torch.uint8)
    fingerprint=prefix_fingerprint(tokens,domains,9)
    packed=SimpleNamespace(tokens=tokens,domains=domains,fingerprint='new-pack',prefix_counts=torch.tensor([[0],[4],[8],[12],[16]]))
    parent={'model':{'w':torch.randn(7)},'optimizer':{'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.randn(7),'exp_avg_sq':torch.rand(7)}},
        'param_groups':[{'params':[0],'lr':.001,'foreach':False,'fused':False}]},
        'cpu_rng':torch.get_rng_state(),'cursor':2,'real_tokens':8,'update':2,'training_matmul_flops':12345,
        'initial_model_digest':'init','pauses':0,'schedule':asdict(WSDSchedule(.001,1)),
        'signature':dict(context=4,seed=23,world=64,batch=2,accumulation=2,optimizer='plain',corpus_sha256='old',packing_sha256=fingerprint)}
    parent['state_digest']=digest({'model':parent['model'],'optimizer':parent['optimizer']})
    metadata={'prefix_parent':{'fingerprint':fingerprint,'source_sha256':'old','windows':2}}
    kwargs=dict(world=32,batch=4,accumulation=2,optimizer='fused',decay_start=12,decay_end=16)
    new,proof=migrate(parent,packed,metadata,'new',**kwargs)
    assert proof['model_optimizer_moment_rng_tensors_unchanged']
    assert new['signature']['world']==32 and new['optimizer']['param_groups'][0]['fused']
    assert not parent['optimizer']['param_groups'][0]['fused']
    assert new['real_tokens']==8 and new['training_matmul_flops']==12345
    parent['schedule']=asdict(WSDSchedule(.001,1,7,10))
    with pytest.raises(ValueError):migrate(parent,packed,metadata,'new',**kwargs)
