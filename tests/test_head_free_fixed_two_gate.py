"""Small numerical/software admission cases,not research data."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from compute_trainium.head_free_fixed_two_gate import validate_report,require_native_qualification,verify_qualified_sources,STATUS
from compute_trainium.lean_baseline_science import update_work,model_factory
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_scientific_schedule import update_work as original_work


def proof():
    # Reuse the real receipt schema,without using its old success as a new gate.
    # Retagged copies below are software fixtures only and never execution plans.
    file=Path(__file__).parents[1]/'results/baseline_numerical_recovery_20261026/report.json'
    old=json.loads(file.read_text())['records']['one']
    record=copy.deepcopy(old)
    for name in ['warm','replay']:
        row=record[name];row['arm']='poisson2';row['qualification_profile']='fixed2';row['managed_qualification_plan_sha256']='new-q-plan'
        row['signature']['preparation_sha256']='new-prep'
        row['signature']['depth_table_sha256']='full-two-table'
    record['independent_cpu']['arm']='poisson2'
    record['restored'].update(status='PASS_BASELINE_NAMED_REPLAY_AND_GLOBAL_GRADIENT',profile='fixed2',plan_sha256='new-q-plan')
    plan=dict(gradient_rule='full_fixed_two',parameters=141963840,depth_table_sha256='full-two-table',
        profiles=dict(fixed2=dict(initial_model_digest=record['warm']['initial_model_digest'],preparation_plan_sha256='new-prep')))
    result=dict(status=STATUS,records={k:copy.deepcopy(record) for k in ['half_a','half_b']},plan_sha256='new-q-plan',
        actual_qualification_optimizer_updates=244,research_optimizer_updates=0,scientific_fit_admitted=False,
        terminal_success=False,test_accessed=False,terminal_alpha_spent=0)
    result['records']['half_a']['warm_physical_chips']=list(range(8))
    result['records']['half_b']['warm_physical_chips']=list(range(8,16))
    return result,plan


def test_full_schema_requires_both_halves_and_cannot_accept_success_label_alone():
    report,plan=proof();assert validate_report(report,plan)
    del report['records']['half_b']
    with pytest.raises(ValueError):validate_report(report,plan)
    report,plan=proof();report['records']['half_a']['independent_cpu']['gradient']['families']['recurrent_injection_readout']['relative_l2']=.021
    with pytest.raises(ValueError):validate_report(report,plan)


def test_first_optimizer_family_and_replay_are_required():
    report,plan=proof();report['records']['half_a']['independent_cpu']['first_actual_update']['families']['muon']['moment_comparison']['relative_l2']=.021
    with pytest.raises(ValueError):validate_report(report,plan)


def test_two_trial_names_do_not_replace_distinct_physical_half_evidence():
    report,plan=proof();report['records']['half_b']['warm_physical_chips']=list(range(8))
    with pytest.raises(ValueError):validate_report(report,plan)
    report,plan=proof();report['records']['half_b']['replay']['per_rank_cpu_rng_on_load_exact']=False
    with pytest.raises(ValueError):validate_report(report,plan)


def test_fixed_two_depth_bytes_and_initialization_cannot_be_substituted():
    report,plan=proof();report['records']['half_a']['warm']['signature']['depth_table_sha256']='sampled-or-truncated'
    with pytest.raises(ValueError):validate_report(report,plan)
    report,plan=proof();report['records']['half_b']['warm']['initial_model_digest']='qualification-updated-state'
    with pytest.raises(ValueError):validate_report(report,plan)


def test_science_role_blocks_before_native_proof(tmp_path):
    with pytest.raises(ValueError):require_native_qualification(tmp_path,{})


def test_fixed_two_nominal_work_is_exact_head_free_full_bptt():
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=128,max_seq=16)
    table=np.full((2,16,32,2),2,dtype=np.int64)
    expected=original_work(c,table,recurrent=True,context=16,world=32,optimizer_per_replica=123)
    actual=update_work('fixed2',c,table,recurrent=True,context=16,world=32,optimizer_per_replica=123)
    np.testing.assert_array_equal(actual,expected)
    for changed in [np.ones_like(table),table.copy()]:
        changed[...,1]=1
        with pytest.raises(ValueError):update_work('fixed2',c,changed,recurrent=True,context=16,world=32,optimizer_per_replica=123)
    from compute_trainium.parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    assert model_factory('fixed2') is ParcaePrimitiveSDPACandidate


def test_new_native_profile_has_explicit_legacy_arm_and_owned_phase():
    from scripts.qualify_baseline_managed import validate_phase
    plan=dict(purpose='baseline_native_qualification_only',root_comm_ports={'fixed-two-half-a-warm':19001},
        phases={'fixed-two-half-a-warm':dict(profile='fixed2',mode='warm')})
    validate_phase(plan,'fixed-two-half-a-warm',19001,profile='fixed2',arm='poisson2',mode='warm')
    with pytest.raises(ValueError):validate_phase(plan,'fixed-two-half-a-warm',19001,profile='fixed2',arm='fixed1',mode='warm')


def test_native_activation_cannot_bypass_a_stopped_main(tmp_path,monkeypatch):
    from scripts import run_head_free_fixed_two_numerical as driver
    monkeypatch.setattr(driver,'managed_running',lambda:False)
    parent=tmp_path/driver.PARENT;parent.mkdir()
    (parent/'report.json').write_text(json.dumps(dict(status='STOPPED_LATE_TWO_ALLOCATION_MAIN',plan_sha256='same')))
    planned=dict(parent_main_plan_sha256='same')
    with pytest.raises(ValueError):driver.activate(tmp_path,tmp_path/'new-plan.json',planned)
    assert not (tmp_path/'activation.json').exists()


def test_native_activation_cannot_overlap_an_existing_owner(tmp_path,monkeypatch):
    from scripts import run_head_free_fixed_two_numerical as driver
    monkeypatch.setattr(driver,'managed_running',lambda:True)
    with pytest.raises(RuntimeError):driver.activate(tmp_path,tmp_path/'new-plan.json',{})


def test_native_snapshot_checks_worker_bytes_without_requiring_host_only_files(tmp_path):
    from compute_trainium.training import file_sha256
    script=tmp_path/'scripts/worker.py';script.parent.mkdir();script.write_text('fixture = True\n')
    worker={'scripts/worker.py':file_sha256(script)}
    host={'docs/host-only.md':'host-file-not-present-in-native-snapshot'}
    plan=dict(source_sha256={**worker,**host},worker_source_sha256=worker,host_only_source_sha256=host)
    verify_qualified_sources(plan,worker_only=True,code_root=tmp_path)
    with pytest.raises(FileNotFoundError):verify_qualified_sources(plan,code_root=tmp_path)
    script.write_text('fixture = False\n')
    with pytest.raises(ValueError):verify_qualified_sources(plan,worker_only=True,code_root=tmp_path)
