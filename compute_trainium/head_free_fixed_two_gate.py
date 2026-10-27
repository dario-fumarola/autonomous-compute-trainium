"""Native proof required before the new head-free fixed-two scientific role."""
import json
import math
from pathlib import Path

from .training import file_sha256
from .native_snapshot_manifest import split_snapshot_sources

STATUS='PASS_HEAD_FREE_FIXED_TWO_NATIVE_QUALIFICATION'
TRIALS={'half_a','half_b'}
GRADIENT_FAMILIES={'recurrent_injection_readout','tied_input_output','transformer_body','value_embeddings'}


def gradient_gate(row):
    if (set(row['families'])!=GRADIENT_FAMILIES or not row['passes_two_percent_gate']
            or any(not math.isfinite(v) or not 0<=v<=.02
                   for v in [row['overall']['relative_l2'],*[r['relative_l2'] for r in row['families'].values()]])):
        raise ValueError('Complete original overall/family gradient gates required')


def optimizer_gate(row,update_key,moment_key):
    values=[row[update_key]['relative_l2'],row[moment_key]['relative_l2']]
    if set(row['families'])!={'adamw','muon'}:raise ValueError('Both named optimizer families required')
    for family in row['families'].values():values.extend([family[update_key]['relative_l2'],family[moment_key]['relative_l2']])
    if any(not math.isfinite(v) or not 0<=v<=.02 for v in values):
        raise ValueError('Original optimizer update/moment gates required')


def validate_report(report,plan):
    if (report['status']!=STATUS or set(report['records'])!=TRIALS
            or report['actual_qualification_optimizer_updates']!=244
            or report['research_optimizer_updates']!=0 or report['scientific_fit_admitted']
            or report['terminal_success'] or report['test_accessed'] or report['terminal_alpha_spent']!=0
            or plan['gradient_rule']!='full_fixed_two' or plan['parameters']!=141963840):
        raise ValueError('Both complete numerical trials with zero scientific admission required')
    if plan.get('rejected_state_evidence_required'):
        from .head_free_fixed_two_v2_gate import fixture_gate
        fixture_gate(report['rejected_state_fixture'],report['plan_sha256'])
    profile=plan['profiles']['fixed2']
    physical=[]
    for record in report['records'].values():
        chips=record['warm_physical_chips']
        if len(chips)!=8 or len(set(chips))!=8 or any(type(c) is not int or not 0<=c<16 for c in chips):
            raise ValueError('Each numerical warm trial must use exactly eight original physical chips')
        physical.extend(chips)
        warm,replay,cpu,restored=(record[k] for k in ['warm','replay','independent_cpu','restored'])
        for mode,native,updates in [('warm',warm,121),('replay',replay,1)]:
            if (native['status']!='CAPTURED_PARCAE_TRAINING_QUALIFICATION' or native['mode']!=mode
                    or native['arm']!='poisson2' or native['precision']!='fp16'
                    or native['world']!=32 or native['accumulation']!=16
                    or native['actual_optimizer_calls_per_replica']!=updates
                    or not native['qualification_only'] or native['research_optimizer_updates']!=0
                    or not native['all_replica_model_optimizer_states_equal']
                    or native['managed_qualification_plan_sha256']!=report['plan_sha256']
                    or native['qualification_profile']!='fixed2'
                    or native['initial_model_digest']!=profile['initial_model_digest']
                    or native['signature']['preparation_sha256']!=profile['preparation_plan_sha256']
                    or native['signature']['depth_table_sha256']!=plan['depth_table_sha256']):
                raise ValueError('Exact full-depth numerical native identity/count/replica gates required')
        if (warm['warm_updates']!=100 or not 0<warm['warm_seconds']<1200
                or not replay['complete_state_on_load_exact'] or not replay['per_rank_cpu_rng_on_load_exact']
                or replay['recovered_state_digest']!=warm['checkpoint_state_digest']
                or cpu['status']!='PASS_INDEPENDENT_TRAINED_PARCAE_GLOBAL_GRADIENT'
                or cpu['arm']!='poisson2' or cpu['windows']!=512 or not cpu['full_state_readback_exact']
                or not math.isfinite(cpu['native_minus_cpu_nll']) or abs(cpu['native_minus_cpu_nll'])>.003):
            raise ValueError('Complete warm measurement,state restoration and full512 independent audit required')
        gradient_gate(cpu['gradient'])
        first=cpu['first_actual_update']
        if not all(first[k] for k in ['passes_registered_gate','independent_group_membership_validated','reference_registration_order_aligned_by_name']):
            raise ValueError('Independent named first-update membership required')
        optimizer_gate(first,'update_comparison','moment_comparison')
        pair=restored['pair']
        if (restored['status']!='PASS_BASELINE_NAMED_REPLAY_AND_GLOBAL_GRADIENT'
                or restored['profile']!='fixed2' or restored['plan_sha256']!=report['plan_sha256']
                or not pair['passes_recovery_gate'] or pair['parameters']!=141963840
                or not pair['complete_state_on_load_exact'] or not pair['per_rank_cpu_rng_on_load_exact']
                or not pair['all_replica_model_optimizer_states_equal']
                or not pair['replay']['passes_two_percent_gate']
                or not math.isfinite(restored['replay_minus_cpu_nll']) or abs(restored['replay_minus_cpu_nll'])>.003):
            raise ValueError('Named replay against the independent CPU gradient required')
        optimizer_gate(pair['replay'],'update','moments');gradient_gate(restored['gradient'])
    if len(set(physical))!=16:raise ValueError('Both distinct physical halves must be covered')
    return True


def verify_qualified_sources(plan,*,worker_only=False,code_root=None):
    worker,host=split_snapshot_sources(plan['source_sha256'])
    if worker!=plan['worker_source_sha256'] or host!=plan['host_only_source_sha256']:
        raise ValueError('Complete host/worker qualification source inventories required')
    code_root=Path(code_root) if code_root is not None else Path(__file__).resolve().parents[1]
    selected=worker if worker_only else plan['source_sha256']
    if not selected or any(file_sha256(code_root/f)!=h for f,h in selected.items()):
        raise ValueError('Qualified native/science source changed')


def require_native_qualification(root,science_plan,*,worker_only=False):
    binding=science_plan.get('fixed_two_native_qualification')
    if not binding:raise ValueError('Head-free fixed-two scientific execution remains unqualified')
    root=Path(root).resolve();file=(root/binding['file']).resolve()
    if not file.is_relative_to(root) or file_sha256(file)!=binding['sha256']:
        raise ValueError('Exact completed private native qualification receipt required')
    report=json.loads(file.read_text());plan_file=file.parent/'plan.json'
    if file_sha256(plan_file)!=report['plan_sha256']:raise ValueError('Exact qualification plan required')
    plan=json.loads(plan_file.read_text());validate_report(report,plan)
    prep=root/plan['profiles']['fixed2']['preparation_directory']
    depth=prep/'poisson2-a16-depths.npy'
    if file_sha256(depth)!=plan['depth_table_sha256']:
        raise ValueError('Qualified full-gradient depth table changed')
    import numpy as np
    values=np.load(depth,mmap_mode='r')
    if values.shape!=(121,16,32,2) or not np.all(values==2):
        raise ValueError('Every actual qualification step must execute and differentiate two passes')
    if plan['model']!=science_plan['model'] or not plan['source_sha256']:
        raise ValueError('Qualified model geometry/source inventory changed')
    verify_qualified_sources(plan,worker_only=worker_only)
    return report
