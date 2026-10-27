import copy

import pytest
import torch

import compute_trainium.rejected_replica_evidence as evidence
from compute_trainium.future_span_scientific_state import validate_checkpoint, SCIENCE
from scripts.qualify_native import digest


def state():
    # Numerical state fixtures only, never source examples or research labels.
    return {'model':{'w':torch.tensor([.1,.2])},
            'optimizer':{'m':torch.tensor([.3,.4]),'step':7}}


def test_no_majority_selection():
    assert evidence.selected_ranks(['a'*64]*32) == ()
    values=['a'*64]*32;values[1]='b'*64
    assert evidence.selected_ranks(values) == (0,1)
    values=['b'*64]+['a'*64]*31
    assert evidence.selected_ranks(values) == (0,1)  # Never substitute the majority.


def test_agreement_and_unselected_rank_never_read_state(tmp_path):
    def forbidden(): raise AssertionError('Unexpected state readback')
    for rank,digests in [(0,['a'*64]*3),(2,['a'*64,'b'*64,'a'*64])]:
        row=evidence.capture_rejected(tmp_path,update=3,rank=rank,digests=digests,
            state_factory=forbidden,digest_fn=digest)
        assert row['status'] in ['AGREEMENT_NO_DIAGNOSTIC','UNSELECTED_REJECTED_REPLICA']
    assert not list(tmp_path.iterdir())


def test_private_rejected_snapshot_preserves_bits_and_cannot_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'require_diagnostic_reserves',lambda *args:None)
    saved=state();before=digest(saved)
    row=evidence.capture_rejected(tmp_path,update=3,rank=1,
        digests=['a'*64,before],state_factory=lambda:saved,digest_fn=digest)
    assert row['status']=='CAPTURED_PRIVATE_REJECTED_STATE' and row['repeated_readback_matches_original_digest']
    payload=torch.load(tmp_path/row['snapshot_file'],weights_only=False)
    assert digest(payload['rejected_complete_state'])==before and digest(saved)==before
    assert payload['purpose']==evidence.PURPOSE and payload['eligible_for_resume'] is False
    with pytest.raises(ValueError,match='purpose'):
        validate_checkpoint(payload,'x',{},dict(purpose=SCIENCE),expected_targets=0,
            expected_domains=[],expected_matrix_ops=0,expected_anchors=0)


def test_snapshot_readback_disagreement_is_preserved_not_certified(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'require_diagnostic_reserves',lambda *args:None)
    row=evidence.capture_rejected(tmp_path,update=3,rank=0,digests=['a'*64,'b'*64],
        state_factory=state,digest_fn=digest)
    assert row['status']=='CAPTURED_PRIVATE_REJECTED_STATE'
    assert not row['repeated_readback_matches_original_digest'] and not row['eligible_for_resume']


def test_reserve_failure_keeps_original_rejection_and_no_tensor_file(tmp_path,monkeypatch):
    def reject(*args):raise RuntimeError('Protected reserve')
    monkeypatch.setattr(evidence,'require_diagnostic_reserves',reject)
    row=evidence.capture_rejected(tmp_path,update=3,rank=0,digests=['a'*64,'b'*64],
        state_factory=state,digest_fn=digest)
    assert row['status']=='FAILED_REJECTED_STATE_CAPTURE'
    assert not list(tmp_path.glob('*.pt')) and list(tmp_path.glob('*.json'))


def test_existing_evidence_is_never_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'require_diagnostic_reserves',lambda *args:None)
    kwargs=dict(update=3,rank=0,digests=['a'*64,'b'*64],state_factory=state,digest_fn=digest)
    first=evidence.capture_rejected(tmp_path,**kwargs)
    file=tmp_path/first['snapshot_file'];before=file.read_bytes()
    second=evidence.capture_rejected(tmp_path,**kwargs)
    assert second['status']=='FAILED_REJECTED_STATE_CAPTURE' and file.read_bytes()==before


def test_inventory_distinguishes_optimizer_model_and_scalar_changes():
    original=state();changed=copy.deepcopy(original);changed['optimizer']['m'][0]+=.000001
    result=evidence.compare_inventories(evidence.leaf_inventory(original),evidence.leaf_inventory(changed))
    assert result['changed_leaf_paths']==["['optimizer']['m']"]
    changed=copy.deepcopy(original);changed['optimizer']['step']=8
    result=evidence.compare_inventories(evidence.leaf_inventory(original),evidence.leaf_inventory(changed))
    assert result['changed_leaf_paths']==["['optimizer']['step']"]


def test_negative_zero_is_visible_in_exact_leaf_hashes():
    left=evidence.leaf_inventory({'x':torch.tensor([0.])})
    right=evidence.leaf_inventory({'x':torch.tensor([-0.])})
    assert evidence.compare_inventories(left,right)['changed_leaf_paths']==["['x']"]


def test_order_only_difference_is_not_mislabeled_as_tensor_difference():
    left=evidence.leaf_inventory({'x':torch.tensor([1.]),'y':7})
    right=evidence.leaf_inventory({'y':7,'x':torch.tensor([1.])})
    row=evidence.compare_inventories(left,right)
    assert row['all_leaf_values_equal'] and not row['structure_equal']


def test_reserves_include_both_snapshots_and_host_memory(tmp_path,monkeypatch):
    class Usage:free=100*evidence.GIB
    monkeypatch.setattr(evidence.shutil,'disk_usage',lambda _:Usage())
    monkeypatch.setattr(evidence,'_available_memory',lambda:600*evidence.GIB)
    monkeypatch.setattr(evidence,'_cgroup_headroom',lambda:20*evidence.GIB)
    evidence.require_diagnostic_reserves(tmp_path,2*evidence.GIB)
    monkeypatch.setattr(evidence,'_available_memory',lambda:512*evidence.GIB)
    with pytest.raises(RuntimeError,match='host reserve'):
        evidence.require_diagnostic_reserves(tmp_path,2*evidence.GIB)
    with pytest.raises(ValueError,match='three-GiB'):
        evidence.require_diagnostic_reserves(tmp_path,4*evidence.GIB)


def test_two_groups_cannot_share_only_one_groups_reservation(tmp_path,monkeypatch):
    class Usage:free=44*evidence.GIB  # Enough for one group, insufficient for two.
    monkeypatch.setattr(evidence.shutil,'disk_usage',lambda _:Usage())
    monkeypatch.setattr(evidence,'_available_memory',lambda:600*evidence.GIB)
    monkeypatch.setattr(evidence,'_cgroup_headroom',lambda:20*evidence.GIB)
    with pytest.raises(RuntimeError,match='disk reserve'):
        evidence.require_diagnostic_reserves(tmp_path,2*evidence.GIB)
    Usage.free=100*evidence.GIB
    monkeypatch.setattr(evidence,'_available_memory',lambda:520*evidence.GIB)
    with pytest.raises(RuntimeError,match='host reserve'):
        evidence.require_diagnostic_reserves(tmp_path,2*evidence.GIB)
    monkeypatch.setattr(evidence,'_available_memory',lambda:600*evidence.GIB)
    monkeypatch.setattr(evidence,'_cgroup_headroom',lambda:6*evidence.GIB)
    with pytest.raises(RuntimeError,match='container memory'):
        evidence.require_diagnostic_reserves(tmp_path,2*evidence.GIB)


def test_unbounded_or_unknown_metadata_cannot_bypass_size_screen():
    with pytest.raises(ValueError,match='metadata string'):
        evidence.tensor_bytes({'x':'a'*(1024**2+1)})
    with pytest.raises(ValueError,match='Unsupported diagnostic metadata'):
        evidence.tensor_bytes({'x':object()})


def test_enforcement_always_rejects_even_when_capture_fails(tmp_path,monkeypatch):
    monkeypatch.setattr(evidence,'capture_rejected',lambda *args,**kwargs:{'status':'FAILED_REJECTED_STATE_CAPTURE'})
    barriers=[]
    with pytest.raises(AssertionError,match='Full scientific replicas differ'):
        evidence.enforce_replica_agreement(tmp_path,update=3,rank=0,digests=['a'*64,'b'*64],
            state_factory=state,digest_fn=digest,barrier=lambda:barriers.append(True),
            failure_message='Full scientific replicas differ')
    assert barriers==[True]


def test_agreement_enforcement_does_not_touch_state_or_barrier(tmp_path):
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected diagnostic activity')
    evidence.enforce_replica_agreement(tmp_path,update=3,rank=0,digests=['a'*64]*3,
        state_factory=forbidden,digest_fn=forbidden,barrier=forbidden,failure_message='fail')
    assert not list(tmp_path.iterdir())
