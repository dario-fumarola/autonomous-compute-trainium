"""Conditional discovery controls;never a terminal confirmation or new recipe."""
import copy
import math

from .future_span_study_plan import CELLS
from .ntp_secondary import ntp_point_contrasts, discovery_decision

IDENTITY = 'future-span-work-control-v1'
QUEUE_IDENTITY = 'future-span-work-control-queue-v1'


def admission_evidence(parent, main_plan, secondary):
    if parent['status'] != 'COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT':
        raise ValueError('All original MAIN work and decisions must be complete')
    if secondary['status'] != 'COMPLETE_NTP_SECONDARY_DISCOVERY_READOUT':
        raise ValueError('Complete registered NTP secondary readout required')
    final = [r for r in parent['scores'] if r['readout'] == 'cooled1b']
    if len(final) != 16 or len({(r['case_id'], r['depth']) for r in final}) != 16:
        raise ValueError('All sixteen unique cooled paths required')
    gates = parent['independent_gradient_gates']
    if len(gates) != 8 or {g['case_id'] for g in gates} != set(main_plan['cases']):
        raise ValueError('All eight unique original numerical gates required')
    for key, case in main_plan['cases'].items():
        rows = [r for r in final if r['case_id'] == key]
        if {r['depth'] for r in rows} != ({1} if case['arm'] == 'one' else {1, 2, 3}):
            raise ValueError('Every declared supported depth must remain present')
        checkpoint = {r['checkpoint_sha256'] for r in rows}
        gate = next(g['independent_cpu'] for g in gates if g['case_id'] == key)
        if (len(checkpoint) != 1 or gate['checkpoint_sha256'] not in checkpoint
                or gate['status'] != 'PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT'
                or gate['training_update'] != case['total_updates']
                or any(r['seed'] != case['seed'] or r['cell'] != case['cell']
                    or r['training_update'] != case['total_updates']
                    or r['training_real_targets'] != case['end_targets']
                    or r['training_parameters'] != case['training_parameters'] for r in rows)):
            raise ValueError('Exact completed endpoint,paired exposure and numerical identity required')
    points = {seed: ntp_point_contrasts([r for r in final if r['seed'] == seed])
              for seed in [5231, 7283]}
    decision = discovery_decision(points)
    observed = {int(k): v for k, v in secondary['points_by_pilot_seed'].items()}
    if observed != points or secondary['secondary_decision'] != decision:
        raise ValueError('Registered secondary decision must reproduce from all original scores')
    return {'eligible': decision['all_point_gates_pass'], 'points': points,
            'unchanged_future_objective_decision': parent['decision'],
            'scope': 'Both discovery seeds required. Not terminal confirmation or an override of the original future-objective result.'}


def validate_control_score(row, case, case_id, checkpoint_sha, plan):
    expected = {'case_id': case_id, 'seed': case['seed'], 'cell': case['cell'],
        'depth': 1, 'training_update': case['total_updates'],
        'training_real_targets': case['end_targets'],
        'training_valid_anchors': case['end_anchors'],
        'estimated_training_dense_matrix_ops': case['training_dense_matrix_ops'],
        'training_parameters': case['training_parameters'], 'checkpoint_sha256': checkpoint_sha,
        'source_corpus_sha256': plan['evaluation']['corpus_sha256'],
        'cohort': 'curve', 'split': 'dev', 'head_removed': True,
        'full_state_readback_exact': True, 'model_unchanged': True, 'test_accessed': False}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError('Complete matched-work readout identity/counters differ')


def make_cases(main_plan, geometry, anchor_prefix):
    origin = geometry['origin_update']
    end = geometry['total_updates']
    preflight = origin + 2
    if not origin < preflight <= geometry['cooldown_start_update'] < end:
        raise ValueError('Two actual continued updates must precede the new cooldown')
    cases = {}
    for seed in [5231, 7283]:
        for cell in ['one-ntp', 'one-future']:
            key, parent = next((k, c) for k, c in main_plan['cases'].items()
                               if c['seed'] == seed and c['cell'] == cell)
            if (parent['arm'] != 'one' or parent['cooldown_start_update'] < origin
                    or parent['cooldown_start_targets'] < geometry['origin_source']['real_targets']):
                raise ValueError('The original parent must still be stable')
            case = copy.deepcopy(parent)
            case.update(parent={'case_id': key, 'update': origin}, source_parent_case_id=key,
                preflight_end_update=preflight, total_updates=end,
                end_targets=geometry['final_source']['real_targets'],
                end_anchors=int(anchor_prefix[geometry['final_source']['source_cursor']]),
                cooldown_start_update=geometry['cooldown_start_update'],
                cooldown_start_targets=geometry['cooldown_source']['real_targets'],
                training_dense_matrix_ops=geometry['matched_work']['executed_matmuls'],
                checkpoint_updates=[preflight, geometry['cooldown_start_update'], end],
                segment_ends=[preflight, end], gradient_updates=[preflight, end],
                primary_inference_depth=1)
            cases[key + '-work'] = case
    return cases


def measured_control_budget(hardware, *, updates=1405, cpu_workers=4):
    if hardware['status'] != 'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION':
        raise ValueError('Complete measured hardware qualification required')
    rows = hardware['records']
    if len(rows) != 4 or {r['case']['identity'] for r in rows} != set(CELLS):
        raise ValueError('All four original measured paths required')
    if type(updates) is not int or updates <= 2 or cpu_workers != 4:
        raise ValueError('Fixed continuation and four CPU workers required')
    durations = []
    rates = {}
    for cell in ['one-ntp', 'one-future']:
        warm = next(r['native']['warm'] for r in rows if r['case']['identity'] == cell)
        if (warm['warm_updates'] != 100 or not 0 < warm['warm_seconds'] <= warm['wall_seconds']
                or not math.isfinite(warm['wall_seconds'])):
            raise ValueError('Actual100 warm updates and finite cold cost required')
        seconds = warm['warm_seconds']/100
        setup = max(120., 1.5*(warm['wall_seconds'] - warm['warm_seconds']))
        # Two scientific segments and two native gradient captures per model.
        duration = updates*seconds*1.5 + 2*setup + 2*180
        durations.extend([duration]*2)
        rates[cell] = {'warm_seconds_per_update': seconds, 'cold_allowance_per_segment': setup}
    loads = [0., 0.]
    for duration in sorted(durations, reverse=True):
        loads[min(range(2), key=lambda i: loads[i])] += duration
    # Separate preflight/final audit waves;do not assume fractional CPU waves.
    cpu = 2*math.ceil(4/cpu_workers)*900 + math.ceil(4/cpu_workers)*360 + 300
    estimate = max(loads) + cpu
    return {'measurement_by_cell': rates, 'cpu_workers': cpu_workers,
        'native_two_half_seconds_allowance': max(loads),
        'cpu_and_transport_seconds_allowance': cpu, 'predicted_seconds_with_margin': estimate,
        'execution_seconds_cap': 3*3600, 'fits_registered_cap': estimate <= 3*3600,
        'assigned_native_chip_hours_allowance': sum(durations)*8/3600,
        'whole_instance_opportunity_chip_hours_cap': 48,
        'scope': 'Prospective same-geometry estimate from actual100 warm updates,50% warm margin,cold launches,eight complete512-gradient audits,four scores and metadata copies. Actual closures and CPU tails replace this estimate;not a chip-time match.'}
