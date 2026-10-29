import copy
import pytest

from compute_trainium.model import ModelConfig
from compute_trainium.chunked_persistent_factory import matrix_work
from scripts.run_chunked_persistent_screen import build_arms,admission_cost,GRID,FAMILIES
from scripts.analyze_chunked_persistent import decisions


class Sources:
    def target_count(self,start,count):
        assert start==0
        return count*511


def original():
    return dict(batch=2,world=64,arms={f'ordinary-lr{i}':dict(lr=lr,
        checkpoint_updates=[1957],warmup_updates=64,depth_schedule=[0]*1957)
        for i,lr in enumerate(GRID)})


def config():
    return ModelConfig(dim=512,layers=11,heads=8,kv_heads=8,ffn_dim=1376,vocab_size=32000,max_seq=512)


def test_work_control_gets_more_authentic_data_and_full_cooldown():
    old=original();before=copy.deepcopy(old);c=config();arms,points=build_arms(old,c,Sources())
    assert old==before and len(arms)==len(points)==9
    work=arms['generated-lr0']['nominal_matrix_flops']
    for i in range(3):
        ordinary=arms[f'ordinary-lr{i}'];generated=arms[f'generated-lr{i}'];constant=arms[f'constant-lr{i}']
        assert generated['checkpoint_updates']==constant['checkpoint_updates']==[1957]
        assert ordinary['checkpoint_updates']==[1969]
        assert ordinary['real_targets']>generated['real_targets']==constant['real_targets']
        assert work==generated['nominal_matrix_flops']==constant['nominal_matrix_flops']
        assert work<=ordinary['nominal_matrix_flops']<work+64*matrix_work(c,512,1,batch=2)['training_matmul_flops']
        for a in (ordinary,generated,constant):
            count=len(a['depth_schedule'])
            assert a['decay_start']==int(.9*count) and a['warmup_updates']==64
            assert a['audit_depths']==list(set(a['depth_schedule']))
    for point in points:
        a=arms[point['arm']]
        assert point['update']==a['checkpoint_updates'][0]
        assert point['variants']==[dict(label='main',depth=a['graph_id'],mode='registered')]


@pytest.mark.parametrize('warm',[{'0':.1,'1':.1,'2':.1},
    {'0':.1,'1':.1,'2':1.01,'3':.1}, {'0':.1,'1':.1,'2':float('nan'),'3':.1}])
def test_unqualified_or_nonfinite_runtime_is_rejected(warm):
    arms,_=build_arms(original(),config(),Sources())
    with pytest.raises(AssertionError):admission_cost(arms,warm)


def test_complete_campaign_cost_includes_other_arms_and_overheads():
    arms,_=build_arms(original(),config(),Sources())
    fast=admission_cost(arms,{str(k):.2 for k in range(4)})
    assert fast['headroom_projected_seconds']>sum(len(a['depth_schedule'])*.2 for a in arms.values())
    with pytest.raises(RuntimeError):admission_cost(arms,{str(k):.9 for k in range(4)})


def test_no_promotion_from_only_one_favorable_or_unbracketed_control():
    selections={f:dict(bracketed=True) for f in FAMILIES}
    comparisons={k:dict(comparison=dict(gain=.02,conditional_95_interval=[.015,.025]))
        for k in ('generated_over_ordinary','generated_over_constant')}
    assert all(decisions(comparisons,selections,.01).values())
    comparisons['generated_over_ordinary']['comparison']['gain']=.005
    assert not decisions(comparisons,selections,.01)['generated_over_ordinary']
    selections['constant']['bracketed']=False
    assert not any(decisions(comparisons,selections,.01).values())
