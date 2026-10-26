"""Admission and analysis for the predeclared late-depth development pair."""
import math
import statistics

WEIGHTS={'web':.6,'math':.2,'code':.2}
SEEDS=(5231,17293)


def gradient_gate(gate,case,update,checkpoint=None):
    native,cpu=gate['native'],gate['independent_cpu']
    gradient=cpu['gradient']
    body=gradient['body']
    errors=[body['overall']['relative_l2'],*[v['relative_l2'] for v in body['families'].values()],
            gradient['future_head']['relative_l2'],*[v['relative_l2'] for v in gradient['future_head_named'].values()]]
    if (native['status']!='CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT'
            or cpu['status']!='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
            or native['case_id']!=case or cpu['case_id']!=case
            or native['training_update']!=update or cpu['training_update']!=update
            or native['checkpoint_sha256']!=cpu['checkpoint_sha256']
            or (checkpoint is not None and cpu['checkpoint_sha256']!=checkpoint)
            or native['windows']!=512 or cpu['windows']!=512
            or not native['replica_gradients_equal'] or not native['model_unchanged']
            or not cpu['full_state_readback_exact'] or not cpu['model_unchanged']
            or native['optimizer_updates']!=0 or cpu['optimizer_updates']!=0
            or not gradient['passes_two_percent_gate'] or not body['passes_two_percent_gate'] or not body['families']
            or any(not math.isfinite(v) or not 0<=v<=.02 for v in errors)
            or set(cpu['loss_differences'])!={'ntp_nll','future_mean'}
            or any(not math.isfinite(v) or abs(v)>.003 for v in cpu['loss_differences'].values())):
        raise ValueError('Exact full512 independent gradient/family/loss/state gate required')
    return True


def qualified_rate(report,plan):
    if report['status']!='COMPLETE_QUALIFIED_LATE_TWO_ALLOCATION_TUNING':
        raise ValueError('Complete TUNE is required;no success-label substitution')
    keys=set(plan['cases'])
    if (len(keys)!=3 or {c['peak_lr'] for c in plan['cases'].values()}!={.008,.016,.032}
            or set(report['scores'])!=keys or set(report['prefix_gates'])!=keys
            or set(report['completed_fit_reports'])!=keys or set(report['actual_warm'])!=keys
            or set(report['switch_readback_checks'])!=keys or report['scientific_optimizer_updates']!=1728):
        raise ValueError('Every registered rate and all counted work must be retained')
    for key,case in plan['cases'].items():
        row=report['scores'][key];warm=report['actual_warm'][key]
        if (row['status']!='COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE'
                or row['case_id']!=key or row['seed']!=4127 or row['training_update']!=576
                or row['depth']!=2 or row['cohort']!='tune' or row['objective_weight']!=0.
                or row['source_corpus_sha256']!=plan['evaluation']['corpus_sha256']
                or row['training_real_targets']!=case['end_targets']
                or row['estimated_training_dense_matrix_ops']!=case['training_dense_matrix_ops']
                or not row['model_unchanged'] or not row['full_state_readback_exact']
                or row['metrics']['weights']!=WEIGHTS or not math.isfinite(row['metrics']['weighted_nll'])):
            raise ValueError('Exact complete cooled TUNE score required for every rate')
        gradient_gate(report['prefix_gates'][key],key,121)
        fit=report['completed_fit_reports'][key];switch=report['switch_readback_checks'][key]
        if (fit['update']!=576 or not fit['complete_state_on_load_exact'] or not fit['cpu_rng_on_load_exact']
                or switch['update']!=432 or not switch['restored_complete_state']
                or not switch['restored_cpu_rng'] or not switch['unchanged_endpoint_schedule']
                or not all(v['all_complete_replica_states_equal'] for v in fit['checkpoint_records'])
                or fit['progress'][-1]['update']!=576 or fit['progress'][-1]['lr']!=0.
                or warm['one_pass_updates']!=100 or warm['two_pass_updates']!=100
                or not 0<warm['one_pass_seconds']<1200 or not 0<warm['two_pass_seconds']<1200):
            raise ValueError('Complete cooldown,restoration,replicas and both real warm regimes required')
    selected=min(keys,key=lambda k:(report['scores'][k]['metrics']['weighted_nll'],plan['cases'][k]['peak_lr']))
    if report['selected_case']!=selected or report['selected_rate']!=plan['cases'][selected]['peak_lr']:
        raise ValueError('Selected TUNE rate disagrees with the complete frozen grid')
    gradient_gate(report['selected_final_gate'],selected,576,report['scores'][selected]['checkpoint_sha256'])
    return report['selected_rate']


def validate_rng_inventory(states):
    import torch
    if not isinstance(states,list) or len(states)!=32:
        raise ValueError('All32 saved per-rank CPU RNG states are required')
    for state in states:
        if not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or state.ndim!=1:
            raise ValueError('Recoverable original CPU RNG tensors required')
        torch.Generator().set_state(state)
    return True


def final_score_gate(row,plan,case_id,checkpoint):
    case=plan['cases'][case_id]
    if (row['status']!='COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE'
            or row['case_id']!=case_id or row['seed']!=case['seed'] or row['depth']!=2
            or row['cohort']!='curve' or row['objective_weight']!=0.
            or row['training_update']!=4400 or row['training_parameters']!=166571840
            or row['deployed_parameters']!=141963840
            or row['checkpoint_sha256']!=checkpoint
            or row['source_corpus_sha256']!=plan['evaluation']['corpus_sha256']
            or row['training_real_targets']!=case['end_targets']
            or row['estimated_training_dense_matrix_ops']!=case['training_dense_matrix_ops']
            or not row['full_state_readback_exact'] or not row['model_unchanged']
            or row['metrics']['weights']!=WEIGHTS or not math.isfinite(row['metrics']['weighted_nll'])):
        raise ValueError('Exact independently scored MAIN endpoint/checkpoint/panel required')
    return True


def bounded_deadline(started,composite_deadline,cap=14400):
    if not (math.isfinite(started) and math.isfinite(composite_deadline) and composite_deadline>started and 0<cap<=14400):
        raise ValueError('Original live composite clock and bounded main cap required')
    return min(started+cap,composite_deadline)


def seed_summary(gains):
    if set(gains)!=set(SEEDS) or not all(math.isfinite(v) for v in gains.values()):
        raise ValueError('Both exact predeclared paired development seeds required')
    values=list(gains.values());mean=statistics.mean(values)
    half=12.706204736432095*statistics.stdev(values)/math.sqrt(2)
    return dict(mean_gain_nats=mean,paired_seed_gains={str(k):v for k,v in gains.items()},
        nominal_paired_seed_interval_95=[mean-half,mean+half],
        development_point_gate=mean>=.01 and all(v>0 for v in values),
        scope='Two reused development seeds on consumed DEV;nominal Student-t df1 interval gives weak variance evidence and does not establish independent confirmation.',
        terminal_success=False)
