import copy
import pytest
import torch

from compute_trainium.late_allocation_recovery import rebind_same_endpoint, original_deadline
from compute_trainium.parcae_scientific_state import schedule_identity


def fixture():
    # Numerical software state, never a generated research example.
    case = dict(peak_lr=.032, warmup_targets=10, cooldown_start_targets=100,
                end_targets=400, total_updates=4, depth_table_sha256='fixed')
    old = dict(plan_sha256='original', world=32, accumulation=16, seed=5231)
    new = {**old, 'plan_sha256':'recovery'}
    saved = dict(signature=old, purpose='natural_future_span_scientific_pilot',
        qualification_only=False, stable_trunk=False, case_id='same', update=3,
        cursor=1536, schedule=schedule_identity(case), real_targets=300,
        model={'w':torch.tensor([.1])}, optimizer={'m':torch.tensor([.2])},
        cpu_rng_by_rank=[torch.get_rng_state()])
    return saved, old, new, case


def test_cooled_recovery_preserves_every_other_object():
    saved, old, new, case = fixture()
    result = rebind_same_endpoint(saved, old, new, case, case, case_id='same', update=3)
    assert result['signature'] == new and saved['signature'] == old
    for key in saved:
        if key != 'signature': assert result[key] is saved[key]


@pytest.mark.parametrize('key,value',[('total_updates',8),('end_targets',800),
    ('peak_lr',.016),('cooldown_start_targets',200),('depth_table_sha256','changed')])
def test_no_new_endpoint_or_scientific_case(key,value):
    saved, old, new, case = fixture(); changed=copy.deepcopy(case); changed[key]=value
    with pytest.raises(ValueError):
        rebind_same_endpoint(saved,old,new,case,changed,case_id='same',update=3)


@pytest.mark.parametrize('key,value',[('stable_trunk',True),('qualification_only',True),
    ('cursor',1024),('case_id','other')])
def test_reject_false_state_metadata(key,value):
    saved,old,new,case=fixture(); saved[key]=value
    with pytest.raises(ValueError):
        rebind_same_endpoint(saved,old,new,case,case,case_id='same',update=3)


def test_reject_signature_science_change():
    saved,old,new,case=fixture(); new['seed']=7
    with pytest.raises(ValueError):
        rebind_same_endpoint(saved,old,new,case,case,case_id='same',update=3)


def test_original_clocks_never_renew():
    assert original_deadline(100,200,300)==200
    assert original_deadline(100,9000,10000)==7300
    with pytest.raises(TimeoutError): original_deadline(201,200,300)
