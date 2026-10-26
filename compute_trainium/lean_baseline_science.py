"""Explicit qualified NTP baseline operators, identities and complete work."""
import numpy as np

from .native_snapshot_manifest import split_snapshot_sources
from .parcae_scientific_schedule import update_work as original_work
from .parcae_scientific_state import signature as original_signature,schedule_identity

ROLES={'one':('fixed1',1),'mixing':('poisson2',2),'ordinary8':('ordinary',1)}
EXECUTION='lean_fp16_primitive_attention_runtime_scalar_baselines'


def evaluation_window_key(document_id,window):
    if not document_id or type(window) is not int or window<0:raise ValueError('Original nonnegative window position required')
    return window


def validate_case(plan,case):
    role=plan.get('model_variant')
    if (role not in ROLES or case['arm']!=ROLES[role][0] or plan['precision']!='fp16'
            or plan['accumulation']!=16 or plan.get('execution_variant')!=EXECUTION
            or case['parameters']!=(141733120 if role=='ordinary8' else 141963840)):
        raise ValueError('Exact qualified lean baseline geometry/variant required')
    return role


def signature(plan_sha,plan,case):
    role=validate_case(plan,case)
    return {**original_signature(plan_sha,plan,case),'model_variant':role,'execution_variant':EXECUTION,
            'optimizer_execution':'qualified_runtime_scalar_muon_fused_adamw',
            'gradient_rule':'all_steps','work_table_sha256':case['work_table_sha256'],'auxiliary_head_present':False}


def worker_sources(plan):
    expected,_=split_snapshot_sources(plan['source_sha256'])
    if ('scripts/train_low_depth_scientific.py' not in expected
            or expected!=plan['worker_source_sha256']):raise ValueError('Exact complete NTP worker source list required')
    return expected


def model_factory(role):
    if role=='mixing':
        from .lean_mixing_precision import LeanFP32MixingCandidate
        return LeanFP32MixingCandidate
    from .parcae_primitive_sdpa_candidate import ParcaePrimitiveSDPACandidate
    return ParcaePrimitiveSDPACandidate


def update_work(role,config,depths,*,recurrent,context,world,optimizer_per_replica):
    values=np.asarray(depths)
    if role not in ROLES or values.ndim!=4 or values.shape[1:]!=(16,world,2) or not np.all(values==ROLES[role][1]):
        raise ValueError('Only the frozen deterministic full-gradient baseline depths are allowed')
    if recurrent!=(role!='ordinary8'):raise ValueError('Baseline recurrence parameterization changed')
    if role!='mixing':return original_work(config,values,recurrent=recurrent,context=context,world=world,optimizer_per_replica=optimizer_per_replica)
    from .parcae_mixing_control import mixing_only_ledger
    neural=mixing_only_ledger(config,context,16*world,total_steps=2,backward_cap=2)['training_matmul_flops']
    total=neural+world*optimizer_per_replica
    if total*len(values)>np.iinfo(np.int64).max:raise OverflowError('Complete baseline work exceeds signed64')
    return np.full(len(values),total,dtype=np.int64)


def make_reference(config,role):
    if role=='ordinary8':
        from scripts.audit_ordinary8_native import make_reference as reference
        return reference(config,False)
    if role=='mixing':
        from .lean_mixing_reference import make_reference as reference
        return reference(config,True)
    from scripts.audit_parcae_precise_training_point import make_reference as reference
    return reference(config,True)


def validate_point(saved,parent_sha,parent,case_id,*,root,parent_file):
    from .training import file_sha256
    import json
    case=parent['cases'][case_id];role=validate_case(parent,case);update=saved['update']
    if (saved['purpose']!=parent['purpose'] or saved['qualification_only'] is not False
            or saved['case_id']!=case_id or saved['signature']!=signature(parent_sha,parent,case)
            or saved['schedule']!=schedule_identity(case) or update not in case['gradient_updates']
            or not 0<update<=case['total_updates'] or saved['cursor']!=update*512):raise ValueError('Exact registered mature baseline point required')
    packing=root/parent['data']['packed'];meta=json.loads((packing/'manifest.json').read_text())
    counts_file=packing/'prefix_counts.npy';work_file=parent_file.parent/case['work_table']
    if (file_sha256(packing/'manifest.json')!=parent['data']['manifest_sha256']
            or file_sha256(counts_file)!=meta['files']['prefix_counts.npy']
            or file_sha256(work_file)!=case['work_table_sha256']):raise AssertionError('Independent baseline counters changed')
    counts=np.load(counts_file,mmap_mode='r');work=np.load(work_file,mmap_mode='r')
    domains=list(map(int,counts[saved['cursor']]-counts[0]))
    if (saved['domain_targets']!=domains or saved['real_targets']!=sum(domains)
            or saved['executed_matmuls']!=int(work[:update].sum())
            or saved['stable_trunk']!=(update<=case['cooldown_start_update'])):raise AssertionError('Exact domain/source/work/cooldown counters differ')
    return case
