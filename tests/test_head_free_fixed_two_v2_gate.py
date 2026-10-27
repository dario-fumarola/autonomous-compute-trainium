import copy
import json
from pathlib import Path

import pytest

from compute_trainium.head_free_fixed_two_v2_gate import (
    FIXTURE_STATUS,fixture_gate,closed_parent_gate,artifact_budget,
)


def fixture():
    return dict(status=FIXTURE_STATUS,plan_sha256='new-numerical-plan',
        purpose='rejected_replica_numerical_fixture_only',world=32,optimizer_updates=0,
        model_forwards=0,research_examples=0,native_tensor_devices_verified=True,
        native_cpu_fixture_bits_match=True,agreement_path_no_state_factory_or_extra_barrier=True,
        deliberate_rejection_verified_all32=True,selected_private_ranks=[0,1],private_snapshot_count=2,
        all_private_snapshot_readbacks_match=True,all_private_snapshots_rejected_by_scientific_validator=True,
        changed_leaf_paths=["['optimizer']['m']"],raw_states_remain_private=True,
        scientific_fit_admitted=False,terminal_alpha_spent=0)


def test_fixture_proof_requires_real_device_rejection_and_private_evidence():
    row=fixture();assert fixture_gate(row,'new-numerical-plan')
    for key in ['native_tensor_devices_verified','native_cpu_fixture_bits_match',
                'deliberate_rejection_verified_all32','all_private_snapshot_readbacks_match',
                'all_private_snapshots_rejected_by_scientific_validator']:
        changed=copy.deepcopy(row);changed[key]=False
        with pytest.raises(ValueError):fixture_gate(changed,'new-numerical-plan')
    for key,value in [('optimizer_updates',1),('private_snapshot_count',32),
        ('selected_private_ranks',[1,2]),('scientific_fit_admitted',True)]:
        changed=copy.deepcopy(row);changed[key]=value
        with pytest.raises(ValueError):fixture_gate(changed,'new-numerical-plan')


def parents():
    # Existing receipt schemas are software fixtures,never newly executed proof.
    directory=Path(__file__).parents[1]/'results/late_two_allocation_20261026'
    read=lambda p:json.loads(p.read_text())
    return (read(directory/'main/report.json'),read(directory/'main/plan.json'),
            read(directory/'single-recovery/report.json'),read(directory/'single-recovery/plan.json'))


def test_explicit_closed_incomplete_parent_chain_is_not_a_quality_gate():
    parent,plan,recovery,recovery_plan=parents()
    assert closed_parent_gate(parent,plan,recovery,recovery_plan)
    recovery['scores']['late-two-ntp-s5231']['metrics']['weighted_nll']=10.
    assert closed_parent_gate(parent,plan,recovery,recovery_plan)
    recovery['original_paired_comparison_complete']=True
    with pytest.raises(ValueError):closed_parent_gate(parent,plan,recovery,recovery_plan)


def test_no_partial_gradient_or_live_parent_can_authorize_new_registration():
    for which in ['status','gradient']:
        parent,plan,recovery,recovery_plan=parents()
        if which=='status':recovery['status']='RUNNING'
        else:recovery['gradient_gates']['late-two-ntp-s5231']['final-gradient']['independent_cpu']['gradient']['body']['families']['transformer_body']['relative_l2']=.021
        with pytest.raises(ValueError):closed_parent_gate(parent,plan,recovery,recovery_plan)


def test_complete_storage_bound_covers_normal_outputs_four_rejections_and_reserve():
    row=artifact_budget(8*1024**3,3*1024**3,141963840)
    assert row['four_rejected_state_allowance']==4*(3*1024**3+64*1024**2)
    assert row['protected_filesystem_reserve']==35*1024**3
    assert row['required_free_nvme_bytes']>35*1024**3+row['four_rejected_state_allowance']+22*1024**3
    with pytest.raises(ValueError):artifact_budget(0,1,141963840)


def test_v2_qualification_receipt_cannot_omit_executed_fixture():
    from tests.test_head_free_fixed_two_gate import proof
    from compute_trainium.head_free_fixed_two_gate import validate_report
    report,plan=proof();plan['rejected_state_evidence_required']=True
    with pytest.raises(KeyError):validate_report(report,plan)
    report['rejected_state_fixture']=fixture();report['rejected_state_fixture']['plan_sha256']='new-q-plan'
    assert validate_report(report,plan)
