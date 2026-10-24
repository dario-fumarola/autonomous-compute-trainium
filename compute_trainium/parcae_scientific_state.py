"""Scientific plan/state boundaries; qualification checkpoints cannot enter."""
from .parcae_scientific_schedule import schedule


def worker_sources(plan):
    expected={p:sha for p,sha in plan['source_sha256'].items()
              if p.startswith(('compute_trainium/','scripts/','configs/'))}
    if ('scripts/train_low_depth_scientific.py' not in expected
        or plan['worker_source_sha256']!=expected):
        raise ValueError('Worker manifest must cover every frozen executable source')
    return expected


def signature(plan_sha,plan,case):
    initial=plan['initial_states'][case['initial_state']]
    return {'plan_sha256':plan_sha,'initial_sha256':initial['sha256'],
            'execution_purpose':plan['purpose'],
            'packing_fingerprint':plan['data']['packing_fingerprint'],
            'source_sha256':plan['data']['corpus_sha256'],
            'model':plan['model'],'arm':case['arm'],'seed':case['seed'],
            'world':32,'context':1024,'accumulation':plan['accumulation'],
            'precision':plan['precision'],'state_seed':case['state_seed'],
            'routing_seed':plan['routing_seed'],'depth_table_sha256':case['depth_table_sha256'],
            'optimizer_recipe':'parcae_muon_adamw_fp32_polar_v2','optimizer_execution':'eager_muon_fused_adamw'}


def schedule_identity(case):
    return {k:case[k] for k in ['peak_lr','warmup_targets','cooldown_start_targets','end_targets','total_updates']}


def validate_checkpoint(saved,case_id,case,expected_signature,*,expected_targets,expected_matmuls):
    purpose=expected_signature.get('execution_purpose','natural_low_depth_scientific_pilot')
    qualification=purpose=='low_depth_driver_integration_qualification'
    if saved['purpose']!=purpose or saved['qualification_only'] is not qualification:
        raise ValueError('Qualification weights cannot start or resume a scientific fit')
    if saved['signature']!=expected_signature:
        raise ValueError('Scientific model/data/execution signature changed')
    if saved['case_id']==case_id:
        if saved['schedule']!=schedule_identity(case):raise ValueError('Same-run recovery changed its schedule')
    else:
        parent=case.get('parent')
        if not parent or saved['case_id']!=parent['case_id'] or saved['update']!=parent['update']:
            raise ValueError('Unregistered checkpoint branch')
        if not saved['stable_trunk']:raise ValueError('A new endpoint cannot extend an annealed branch')
        # Parent and child share the absolute warmup and peak rate before decay.
        if any(saved['schedule'][k]!=case[k] for k in ['peak_lr','warmup_targets']):
            raise ValueError('Branch changed the shared-prefix learning recipe')
    if saved['cursor']!=saved['update']*32*expected_signature['accumulation'] or saved['update']>=case['total_updates']:
        raise ValueError('Scientific cursor/update/endpoint mismatch')
    if saved['real_targets']!=expected_targets or saved['executed_matmuls']!=expected_matmuls:
        raise ValueError('Scientific source or executed-work counter mismatch')


def configure_optimizer(optimizer,case,after_update_targets):
    row=schedule(after_update_targets,warmup_targets=case['warmup_targets'],
                 cooldown_start_targets=case['cooldown_start_targets'],end_targets=case['end_targets'],
                 peak_lr=case['peak_lr'])
    for group in optimizer.param_groups:
        group['lr']=row['lr']*group.get('base_lr',1.)
        if group['kind']=='muon':
            group['momentum']=row['muon_momentum'];group['weight_decay']=row['muon_cautious_decay']
    return row
