"""Scientific objective identity,counters and stable-prefix branch boundaries."""
import numpy as np

from .future_span import FutureSpanConfig
from .parcae_scientific_schedule import update_work
from .parcae_scientific_state import schedule_identity


SCIENCE='natural_future_span_scientific_pilot'
INTEGRATION='future_span_driver_integration_qualification'


def matrix_work(config,span,depths,*,world,optimizer_per_replica):
    """Estimated dense semantic matrix work,not physical/tile/network FLOPs.

    Both objective weights execute the same nominal head autograd path. Count
    every head slot,including invalid spans,and all replicas' optimizer work.
    Actual pruning,pointwise kernels and communication are reported separately.
    """
    values=np.asarray(depths)
    if values.ndim!=4 or values.shape[-1]!=2 or not np.array_equal(values[...,0],values[...,1]):
        raise ValueError('Every sampled recurrence step must receive gradients')
    base=update_work(config,values,recurrent=True,context=span.context,world=world,
                     optimizer_per_replica=optimizer_per_replica)
    extra=values.shape[1]*world*6*len(span.positions)*config.dim*config.vocab_size
    if int(base.max())+extra>np.iinfo(np.int64).max:raise OverflowError('Auxiliary work exceeds signed64 ledger')
    return base+extra


def worker_sources(plan):
    expected={p:sha for p,sha in plan['source_sha256'].items()
              if p.startswith(('compute_trainium/','scripts/','configs/','third_party/'))}
    if ('scripts/train_future_span_scientific.py' not in expected or plan['worker_source_sha256']!=expected):
        raise ValueError('Exact complete saved executable source list required')
    return expected


def signature(plan_sha,plan,case):
    if plan['purpose'] not in [SCIENCE,INTEGRATION]:raise ValueError('Explicit future-span scientific/driver purpose required')
    if case['arm'] not in ['one','recurrent'] or case['objective_weight'] not in [0.,.1]:
        raise ValueError('Only the declared four cells are supported')
    if case['cell']!=case['arm']+('-future' if case['objective_weight'] else '-ntp'):
        raise ValueError('Cell label disagrees with its actual model/objective')
    span=FutureSpanConfig(**plan['future_config'])
    if span.weight!=.1 or plan['gradient_rule']!='all_sampled_steps':raise ValueError('Frozen objective and full-gradient rule required')
    initial=plan['initial_states'][case['initial_state']]
    return {'schema':'future-span-scientific-state-v1','plan_sha256':plan_sha,
        'purpose':plan['purpose'],'initial_sha256':initial['sha256'],'model':plan['model'],
        'future_config':plan['future_config'],'cell':case['cell'],'arm':case['arm'],
        'objective_weight':case['objective_weight'],'seed':case['seed'],'state_seed':case['state_seed'],
        'world':32,'context':span.context,'accumulation':plan['accumulation'],
        'precision':plan['precision'],'gradient_rule':plan['gradient_rule'],'routing_seed':plan['routing_seed'],
        'depth_table_sha256':case['depth_table_sha256'],'work_table_sha256':case['work_table_sha256'],
        'packing_fingerprint':plan['data']['packing_fingerprint'],'corpus_sha256':plan['data']['corpus_sha256'],
        'anchor_prefix_sha256':plan['auxiliary_data']['prefix_sha256'],
        'optimizer_recipe':'parcae_muon_adamw_fp32_polar_v2','optimizer_execution':'eager_muon_fused_adamw'}


def validate_initial(saved,plan,case):
    qualification=plan['purpose']==INTEGRATION
    if saved['qualification_only'] is not qualification:
        raise ValueError('Qualification initialization cannot enter a scientific fit')
    seed=saved.get('seed',saved.get('backbone_model_seed'))
    if seed!=case['seed'] or saved['config']!=plan['model'] or saved['future_config']!=plan['future_config']:
        raise ValueError('Fresh seed/body/auxiliary initializer differs')


def validate_checkpoint(saved,case_id,case,expected_signature,*,expected_targets,expected_domains,
                        expected_matrix_ops,expected_anchors):
    qualification=expected_signature['purpose']==INTEGRATION
    if (saved['purpose']!=expected_signature['purpose'] or saved['qualification_only'] is not qualification
            or saved['signature']!=expected_signature):
        raise ValueError('Scientific checkpoint purpose/model/objective/data/execution changed')
    if saved['stable_trunk']!=(saved['real_targets']<=saved['schedule']['cooldown_start_targets']):
        raise ValueError('Checkpoint falsely labels an annealed state as stable')
    if saved['case_id']==case_id:
        if saved['schedule']!=schedule_identity(case):raise ValueError('Resume changed the declared schedule')
    else:
        parent=case.get('parent')
        if not parent or saved['case_id']!=parent['case_id'] or saved['update']!=parent['update']:
            raise ValueError('Unregistered scientific branch')
        if not saved['stable_trunk'] or saved['real_targets']>case['cooldown_start_targets']:
            raise ValueError('A new endpoint must branch before either cooldown')
        if any(saved['schedule'][k]!=case[k] for k in ['peak_lr','warmup_targets']):
            raise ValueError('Branch changed its shared-prefix learning recipe')
    if saved['cursor']!=saved['update']*32*expected_signature['accumulation'] or not 0<saved['update']<case['total_updates']:
        raise ValueError('Source cursor/update/endpoint mismatch')
    if (saved['real_targets']!=expected_targets or saved['domain_targets']!=list(expected_domains)
            or saved['estimated_dense_matrix_ops']!=expected_matrix_ops or saved['valid_anchors']!=expected_anchors):
        raise ValueError('Source/domain/matrix/auxiliary counters differ')
    span=FutureSpanConfig(**expected_signature['future_config'])
    if (saved['future_source_position_presentations']!=expected_anchors*(span.horizon-1)
            or saved['executed_head_slots']!=saved['cursor']*len(span.positions)):
        raise ValueError('Future source-label presentations and paid head slots differ')
