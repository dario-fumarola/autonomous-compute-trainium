import copy
import pytest
import torch

from compute_trainium.future_span_recovery import rebind_stable_checkpoint
from compute_trainium.parcae_scientific_state import schedule_identity


def fixture():
    case=dict(peak_lr=.032,warmup_targets=50,cooldown_start_targets=100,
              end_targets=200,total_updates=2,final_lr_fraction=0.)
    # Numerical state fixtures only; never source examples or research labels.
    old=dict(plan_sha256='original',purpose='natural_future_span_scientific_pilot',seed=23)
    new={**old,'plan_sha256':'registered-recovery'}
    state=dict(signature=old,purpose=old['purpose'],qualification_only=False,stable_trunk=True,
               update=1,cursor=512,schedule=schedule_identity(case),real_targets=100,
               model={'w':torch.tensor([.1])},optimizer={'moment':torch.tensor([.2])},
               cpu_rng_by_rank=[torch.get_rng_state()],estimated_dense_matrix_ops=123)
    return state,old,new,case


def test_rebind_changes_only_metadata_and_keeps_tensor_objects():
    saved,old,new,case=fixture();result=rebind_stable_checkpoint(saved,old,new,case,update=1)
    assert result['signature']==new and saved['signature']==old
    for key in saved:
        if key!='signature':assert result[key] is saved[key]


@pytest.mark.parametrize('field,value',[('stable_trunk',False),('qualification_only',True),
    ('real_targets',101),('cursor',1024),('update',2)])
def test_reject_changed_or_annealed_prefix(field,value):
    saved,old,new,case=fixture();saved[field]=value
    with pytest.raises(ValueError):rebind_stable_checkpoint(saved,old,new,case,update=1)


def test_reject_scientific_change_disguised_as_recovery():
    saved,old,new,case=fixture();new=copy.deepcopy(new);new['seed']=211
    with pytest.raises(ValueError):rebind_stable_checkpoint(saved,old,new,case,update=1)
