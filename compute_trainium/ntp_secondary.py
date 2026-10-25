"""Prospective secondary discovery;never overrides the factorial primary claim."""
import math


def ntp_point_contrasts(rows):
    wanted={('one-ntp',1),('one-future',1),('recurrent-ntp',1),('recurrent-ntp',2)}
    chosen={(r['cell'],r['depth']):r for r in rows if (r['cell'],r['depth']) in wanted}
    if set(chosen)!=wanted or sum((r['cell'],r['depth']) in wanted for r in rows)!=4:
        raise ValueError('Exactly the four declared trained paths are required')
    if (len({r['seed'] for r in chosen.values()})!=1
            or len({r['training_real_targets'] for r in chosen.values()})!=1
            or len({r['training_parameters'] for r in chosen.values()})!=1
            or len({r['deployed_parameters'] for r in chosen.values()})!=1):
        raise ValueError('Paired seed,source exposure and parameter inventory required')
    if chosen['recurrent-ntp',1]['checkpoint_sha256']!=chosen['recurrent-ntp',2]['checkpoint_sha256']:
        raise ValueError('Supported depth intervention must use identical weights')
    loss={key:r['metrics']['weighted_nll'] for key,r in chosen.items()}
    if any(not math.isfinite(value) for value in loss.values()):raise ValueError('Finite prediction losses required')
    return {'over_one_ntp':loss['one-ntp',1]-loss['recurrent-ntp',2],
            'over_one_future':loss['one-future',1]-loss['recurrent-ntp',2],
            'supported_depth1_to2':loss['recurrent-ntp',1]-loss['recurrent-ntp',2]}


def discovery_decision(per_seed,threshold=.01):
    if set(per_seed)!={5231,7283}:raise ValueError('Both frozen pilot seeds required')
    required={'over_one_ntp','over_one_future','supported_depth1_to2'}
    if any(set(row)!=required or any(not math.isfinite(value) for value in row.values())
           for row in per_seed.values()):raise ValueError('Every declared finite contrast is required')
    passed=all(value>=threshold for row in per_seed.values() for value in row.values())
    return {'decision':'NTP_CANDIDATE_FOR_STRONG_BASELINE_RESOURCE_CONTROLS_AND_FRESH_CONFIRMATION' if passed
            else 'NO_USEFUL_NTP_SECONDARY_CANDIDATE_AT_THIS_ENDPOINT',
            'all_point_gates_pass':passed,'minimum_useful_gain_nats':threshold,
            'terminal_success':False,'scope':'Adaptive TUNE-selected secondary discovery,not independent confirmation or a primary future-objective success.'}
