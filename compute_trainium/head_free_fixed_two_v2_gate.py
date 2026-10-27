"""New numerical registration after closed,incomplete late-two research."""
import json
from pathlib import Path

from .late_allocation_main import gradient_gate,final_score_gate
from .training import file_sha256

FIXTURE_STATUS='PASS_NATIVE_REJECTED_REPLICA_EVIDENCE_FIXTURE'


def fixture_gate(row,plan_sha):
    if (row['status']!=FIXTURE_STATUS or row['plan_sha256']!=plan_sha
            or row['purpose']!='rejected_replica_numerical_fixture_only'
            or row['world']!=32 or row['optimizer_updates']!=0 or row['model_forwards']!=0
            or not row['native_tensor_devices_verified'] or not row['native_cpu_fixture_bits_match']
            or not row['agreement_path_no_state_factory_or_extra_barrier']
            or not row['deliberate_rejection_verified_all32']
            or row['selected_private_ranks']!=[0,1] or row['private_snapshot_count']!=2
            or not row['all_private_snapshot_readbacks_match']
            or not row['all_private_snapshots_rejected_by_scientific_validator']
            or row['changed_leaf_paths']!=["['optimizer']['m']"]
            or not row['raw_states_remain_private'] or row['scientific_fit_admitted']
            or row['research_examples']!=0 or row['terminal_alpha_spent']!=0):
        raise ValueError('Actual zero-update Native agreement/rejection/private-evidence proof required')
    return True


def closed_parent_gate(parent,parent_plan,recovery,recovery_plan):
    if (parent['status']!='STOPPED_LATE_TWO_ALLOCATION_MAIN'
            or recovery['status']!='COMPLETE_AUDITED_DESCRIPTIVE_LATE_TWO_SINGLE_RECOVERY'
            or recovery['original_paired_comparison_complete'] is not False
            or recovery['further_control_preparation_eligible'] is not False
            or recovery['fresh_confirmation_admitted'] is not False
            or set(recovery_plan['cases'])!={'late-two-ntp-s5231'}
            or set(recovery['scores'])!={'late-two-ntp-s5231'}):
        raise ValueError('Preserved stopped pair and complete audited single endpoint required;no success substitution')
    for seed in [5231,17293]:
        key=f'late-two-ntp-s{seed}'
        gradient_gate(parent['gradient_gates'][key]['prefix-gradient'],key,121)
    key='late-two-ntp-s5231'
    for label,update in [('switch-gradient',3300),('final-gradient',4400)]:
        gradient_gate(recovery['gradient_gates'][key][label],key,update)
    score=recovery['scores'][key]
    final_score_gate(score,recovery_plan,key,score['checkpoint_sha256'])
    if (score['checkpoint_sha256']!=recovery['gradient_gates'][key]['final-gradient']['independent_cpu']['checkpoint_sha256']
            or not recovery['state_readbacks'][key]['4400']['all32_rng_states_recoverable']
            or not recovery['state_readbacks'][key]['4400']['replicas_equal']
            or parent_plan['model']!=recovery_plan['model']):
        raise ValueError('Complete exact state/score/final audit chain required')
    return True


def parent_binding(root):
    root=Path(root)
    names=['late-two-allocation-main-v1/plan.json','late-two-allocation-main-v1/report.json',
           'late-two-single-recovery-v1/plan.json','late-two-single-recovery-v1/report.json']
    values=[json.loads((root/p).read_text()) for p in names]
    parent_plan,parent,recovery_plan,recovery=values
    if (parent['plan_sha256']!=file_sha256(root/names[0])
            or recovery['plan_sha256']!=file_sha256(root/names[2])
            or recovery_plan['original_parent_plan_sha256']!=parent['plan_sha256']
            or recovery_plan['original_parent_report_sha256']!=file_sha256(root/names[1])):
        raise ValueError('Exact immutable closed-parent plan/report hashes required')
    closed_parent_gate(parent,parent_plan,recovery,recovery_plan)
    return {name:file_sha256(root/name) for name in names}


def artifact_budget(warm_bytes,replay_bytes,parameters):
    if not all(type(v) is int and v>0 for v in [warm_bytes,replay_bytes,parameters]):
        raise ValueError('Actual complete matching-model artifact sizes required')
    cpu=parameters*4+64*1024**2
    normal=int(2*(warm_bytes+replay_bytes+cpu)*1.25)
    rejected=4*(3*1024**3+64*1024**2)
    return dict(two_trial_normal_artifacts_with_margin=normal,
        four_rejected_state_allowance=rejected,nvme_compiler_snapshot_allowance=4*1024**3,
        protected_filesystem_reserve=35*1024**3,
        required_free_nvme_bytes=normal+rejected+4*1024**3+35*1024**3,
        required_free_other_filesystem_bytes=36*1024**3,
        scope='Two actual matching head-free model warm/replay inventories,25% normal margin,full four-state diagnostic maximum,4GiB NVMe compiler/snapshot allowance;35GiB reserves unchanged. Other mounted filesystems reserve an additional1GiB for observed small log/overlay growth.')
