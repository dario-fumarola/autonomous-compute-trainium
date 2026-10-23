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
