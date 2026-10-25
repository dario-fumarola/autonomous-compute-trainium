"""Read-only prediction-path checks and a bounded prefill timing budget."""
import math
import numpy as np
import torch

from .metrics import real_token_losses


def readout(logits, targets, real_mask):
    if logits.shape[1] % 8:
        raise ValueError('Fixed context must have eight evenly spaced logit checks')
    return (real_token_losses(logits, targets, real_mask).sum(),
            logits[:, ::logits.shape[1]//8].contiguous())


def check_readout(loss_sum, samples, expected_sum, expected_samples, real_targets):
    if real_targets < 1 or samples.shape != expected_samples.shape:
        raise ValueError('Identical nonempty source targets and checked logits required')
    actual, reference = samples.double(), expected_samples.double()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite inference logits')
    delta = float(loss_sum)-float(expected_sum)
    relative = float(torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-30))
    result = {'nll_difference': delta/real_targets, 'logit_relative_l2': relative,
        'maximum_logit_error': float((actual-reference).abs().max()),
        'passes': math.isfinite(delta) and abs(delta/real_targets) <= .003 and relative <= .02}
    if not result['passes']:
        raise ValueError('Unchanged0.003-nat/2% numerical gates failed: ' + str(result))
    return result


def prefill_budget(hardware):
    if hardware['status'] != 'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION':
        raise ValueError('Actual same-geometry100-warm-update evidence required')
    by_cell = {r['case']['identity']:r['native']['warm'] for r in hardware['records']}
    if set(by_cell) != {'one-ntp','one-future','recurrent-ntp','recurrent-future'}:
        raise ValueError('All four qualified paths required')
    if any(x['warm_updates'] != 100 or not 0 < x['warm_seconds'] <= x['wall_seconds'] for x in by_cell.values()):
        raise ValueError('Complete100-warm-update measurements required')
    # No inference speedup is assumed:four times the measured full training
    # microstep,plus twice cold training setup and model-state checking.
    native = {}
    for cell, exits in [('one-ntp',1),('recurrent-ntp',3)]:
        row = by_cell[cell]
        call = 4*row['warm_seconds']/100/16
        cold = 2*(row['wall_seconds']-row['warm_seconds'])
        native[cell] = {'forward_seconds_allowance':call,'cold_seconds_allowance':cold,
            'phase_seconds_allowance':exits*204*call+cold+30}
    estimate = 2*max(x['phase_seconds_allowance'] for x in native.values())+300+60
    return {'by_cell':native,'two_rounds_plus_cpu_seconds_allowance':estimate,
        'execution_seconds_cap':1800,'fits_cap':bool(np.isfinite(estimate) and estimate <= 1800
            and max(v['phase_seconds_allowance'] for v in native.values()) <= 660),
        'cpu_reference_seconds_cap':300,'native_phase_seconds_cap':660,
        'scope':'Prospective feasibility bound from measured training,not measured inference. Four native phases,two concurrent halves per round;100 warm and100 timed calls plus two pre/post checks per supported depth. Setup,CPU reference and tails are charged separately.'}


def validate_phase(row, plan, case_id, round_number, plan_sha):
    from .parcae_candidate import ParcaeCandidateConfig, matmul_ledger
    case=plan['cases'][case_id];config=ParcaeCandidateConfig(**plan['model'])
    expected={'status':'PASS_READ_ONLY_NATIVE_PREFILL_TIMING','case_id':case_id,'round':round_number,
        'plan_sha256':plan_sha,'checkpoint_sha256':case['checkpoint_sha256'],
        'deployed_parameters':config.parameter_count(True),'head_removed':True,'weights_unchanged':True,
        'unique_windows':64,'context':1024,'world':32,'optimizer_updates':0,
        'gradients_created':False,'test_accessed':False}
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Read-only timed phase identity differs')
    if row['physical_chips'] not in [list(range(8)),list(range(8,16))]:
        raise ValueError('Exactly one independently owned eight-chip half required')
    if len(row['records'])!=len(case['depths']) or {r['depth'] for r in row['records']}!=set(case['depths']):
        raise ValueError('All and only registered supported depths required')
    for r in row['records']:
        if (r['warm_forwards_per_rank']!=100 or r['timed_forwards_per_rank']!=100 or r['actual_forwards']!=204*32
                or r['forward_matrix_ops_per_call']!=matmul_ledger(config,1024,total_steps=r['depth'])['forward_matmul_flops']
                or not 0 < r['timed_forward_median_seconds'] <= r['timed_forward_p95_seconds']
                or not 0 <= r['maximum_absolute_nll_error'] <= .003 or not 0 <= r['maximum_logit_relative_l2'] <= .02):
            raise ValueError('Complete measured calls,work or numerical gates differ')
