import copy
import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from compute_trainium.future_span import FutureSpanCandidate,FutureSpanConfig,future_span_targets
from compute_trainium.future_span_scientific_state import SCIENCE,signature,validate_initial,validate_checkpoint,matrix_work
from compute_trainium.future_span_training import FutureSpanObjective,objective_coefficients,training_window
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_scientific_state import schedule_identity
from compute_trainium.training import PackedCorpus


def setup():
    future={'context':1024,'horizon':32,'stride':16,'weight':.1}
    plan={'purpose':SCIENCE,'model':{'dim':768},'future_config':future,'gradient_rule':'all_sampled_steps',
        'initial_states':{'fresh':{'sha256':'initial'}},'data':{'packing_fingerprint':'pack','corpus_sha256':'corpus'},
        'auxiliary_data':{'prefix_sha256':'anchors'},'precision':'fp16_body_fp32_heads_masters',
        'accumulation':16,'routing_seed':271828}
    case={'cell':'recurrent-future','arm':'recurrent','objective_weight':.1,'seed':5231,'state_seed':7240,
        'initial_state':'fresh','depth_table_sha256':'depth','work_table_sha256':'work','total_updates':40,
        'peak_lr':.016,'warmup_targets':100,'cooldown_start_targets':1000,'end_targets':2000}
    sig=signature('plan',plan,case)
    saved={'purpose':SCIENCE,'qualification_only':False,'signature':sig,'case_id':'parent',
        'schedule':schedule_identity(case),'stable_trunk':True,'update':20,'cursor':20*512,
        'real_targets':1000,'domain_targets':[600,200,200],'estimated_dense_matrix_ops':123456,
        'valid_anchors':111,'future_source_position_presentations':111*31,'executed_head_slots':20*512*63}
    return plan,case,sig,saved


def validate(saved,key,case,sig):
    return validate_checkpoint(saved,key,case,sig,expected_targets=1000,expected_domains=[600,200,200],
        expected_matrix_ops=123456,expected_anchors=111)


def test_stable_prefix_branch_preserves_objective_optimizer_history_and_every_counter():
    plan,parent,sig,saved=setup()
    validate(saved,'parent',parent,sig)
    child={**parent,'total_updates':30,'end_targets':1500,'parent':{'case_id':'parent','update':20}}
    validate(saved,'child',child,sig)
    for field in ['valid_anchors','estimated_dense_matrix_ops','future_source_position_presentations','executed_head_slots','cursor']:
        bad=copy.deepcopy(saved);bad[field]+=1
        with pytest.raises(ValueError):validate(bad,'parent',parent,sig)
    bad=copy.deepcopy(saved);bad['real_targets']=1001
    with pytest.raises(ValueError):validate(bad,'child',child,sig)
    bad=copy.deepcopy(saved);bad['qualification_only']=True
    with pytest.raises(ValueError):validate(bad,'parent',parent,sig)
    for field,value in [('objective_weight',0.),('state_seed',7),('work_table_sha256','other')]:
        changed={**parent,field:value}
        with pytest.raises(ValueError):validate(saved,'parent',changed,signature('plan',plan,changed))
    with pytest.raises(ValueError):validate(saved,'child',{**child,'peak_lr':.032},sig)
    validate_initial({'qualification_only':False,'seed':5231,'config':plan['model'],
                      'future_config':plan['future_config']},plan,parent)
    with pytest.raises(ValueError):validate_initial({'qualification_only':True},plan,parent)


def test_complete_future_head_matrix_count_matches_actual_autograd_even_at_zero_weight():
    config=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=64,max_seq=16)
    span=FutureSpanConfig(context=16,horizon=4,stride=3)
    docs=[{'id':'fixture','domain':'web','split':'train','tokens':list(range(3,32))}]
    packed=PackedCorpus(docs,16);model=FutureSpanCandidate(config,span,fp16_body=False)
    objective=FutureSpanObjective(model);values=training_window(packed,0,config,span,99)
    count=int(values[2].sum());anchors=int(values[-1].sum())
    for n,weight in [(1,0.),(2,.1)]:
        model.zero_grad()
        with sdpa_kernel(SDPBackend.MATH),FlopCounterMode(display=False) as counter:
            loss,_=objective(*values,objective_coefficients(count,anchors,weight,1),n,n);loss.backward()
        table=np.array([[[[n,n]]]],dtype=np.uint16)
        estimate=matrix_work(config,span,table,world=1,optimizer_per_replica=0)
        assert int(estimate[0])==counter.get_total_flops()
    with pytest.raises(ValueError):matrix_work(config,span,np.array([[[[2,1]]]]),world=1,optimizer_per_replica=0)
