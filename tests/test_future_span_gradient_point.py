import copy
import pytest

from compute_trainium.future_span_gradient_point import validate_point
from compute_trainium.future_span_scientific_state import SCIENCE,signature
from compute_trainium.parcae_scientific_state import schedule_identity


def test_registered_mature_endpoint_has_full_identity_and_auxiliary_work_counters():
    case={'cell':'one-future','arm':'one','objective_weight':.1,'seed':5231,'state_seed':7240,
        'initial_state':'fresh','depth_table_sha256':'depth','work_table_sha256':'work','total_updates':1920,
        'gradient_updates':[576,1920],'peak_lr':.016,'warmup_targets':50,'cooldown_start_targets':500,'end_targets':1000}
    plan={'purpose':SCIENCE,'model':{},'future_config':{'context':1024,'horizon':32,'stride':16,'weight':.1},
        'initial_states':{'fresh':{'sha256':'initial'}},'gradient_rule':'all_sampled_steps',
        'data':{'packing_fingerprint':'pack','corpus_sha256':'source'},'auxiliary_data':{'prefix_sha256':'anchors'},
        'accumulation':16,'precision':'fp16_body_fp32_heads_masters','routing_seed':2,'cases':{'one':case}}
    saved={'purpose':SCIENCE,'qualification_only':False,'case_id':'one','signature':signature('plan',plan,case),
        'schedule':schedule_identity(case),'update':1920,'cursor':1920*512,'domain_targets':[600,200,200],
        'real_targets':1000,'valid_anchors':300,'estimated_dense_matrix_ops':1234,
        'future_source_position_presentations':300*31,'executed_head_slots':1920*512*63,'stable_trunk':False}
    assert validate_point(saved,'plan',plan,'one',[600,200,200],300,1234)==case
    for field,value in [('qualification_only',True),('purpose','qualification'),('update',1919),
                        ('valid_anchors',301),('stable_trunk',True),('estimated_dense_matrix_ops',1235),
                        ('future_source_position_presentations',9301),('executed_head_slots',1),('cursor',1)]:
        changed=copy.deepcopy(saved);changed[field]=value
        with pytest.raises(ValueError):validate_point(changed,'plan',plan,'one',[600,200,200],300,1234)
