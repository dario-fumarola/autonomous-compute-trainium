"""Predeclared integer compute endpoint and paired confirmation intervals."""
import math
import statistics


def matched_endpoint(reference_steps, reference_flops, candidate_flops):
    if min(reference_steps, reference_flops, candidate_flops) <= 0:
        raise ValueError('Positive steps and per-step work are required')
    steps = reference_steps * reference_flops // candidate_flops
    return {'updates': steps, 'stable_update': steps * 9 // 10,
            'reference_flops': reference_steps * reference_flops,
            'candidate_flops': steps * candidate_flops,
            'rounding_shortfall_flops': reference_steps * reference_flops - steps * candidate_flops}


def paired_seed_interval(gains):
    critical = {2: 12.706204736432095, 3: 4.302652729911275}
    if len(gains) not in critical or not all(math.isfinite(x) for x in gains.values()):
        raise ValueError('Require two or three finite paired training-seed gains')
    mean = statistics.mean(gains.values())
    half = critical[len(gains)] * statistics.stdev(gains.values()) / math.sqrt(len(gains))
    lower, upper = mean-half, mean+half
    return {'mean_gain_nats': mean, 'interval_95': [lower, upper],
            'paired_seed_gains': gains, 'degrees_of_freedom': len(gains)-1,
            'decision': 'USEFUL_GAIN_SUPPORTED' if lower > .01 else
                ('USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE' if upper < .01 else 'INCONCLUSIVE'),
            'assumption': 'Independent paired training seeds, approximately normal differences; threshold 0.01 nats.'}


def endpoint_inventory(container_root, flop_update=6496):
    models=[]
    for seed in [23,47,71]:
        for arm in ['ordinary','shared','untied']:
            if arm=='ordinary':prefix=f'/work/runs/natural-curve-v1-ordinary-s{seed}-u3840'
            elif arm=='shared':prefix=f'{container_root}/runs/preln-recurrent-v1-s{seed}-u3840'
            elif seed==23:prefix=f'{container_root}/runs/untied-depth-v1-s23-u3840'
            else:prefix=f'{container_root}/runs/confirmation-v3-untied-s{seed}-u3840'
            models.append({'identity':f'{arm}-1b-s{seed}','arm':arm,'seed':seed,'endpoint':'1B',
                'checkpoint':prefix+'/checkpoint.pt','reference':prefix+'-dev/report.json','passes':1 if arm=='ordinary' else 2})
    for arm,tag in [('ordinary','ordinary'),('shared','recurrent')]:
        prefix=f'{container_root}/runs/stable-cont-v2-{tag}-s23-u11520'
        models.append({'identity':f'{arm}-3b-s23','arm':arm,'seed':23,'endpoint':'3B',
            'checkpoint':prefix+'/checkpoint.pt','reference':prefix+'-dev/report.json','passes':1 if arm=='ordinary' else 2})
    prefix=f'{container_root}/runs/confirmation-v3-ordinary-flop-s23-u{flop_update}'
    models.append({'identity':'ordinary-flop-s23','arm':'ordinary','seed':23,'endpoint':'matched FLOPs',
        'checkpoint':prefix+'/checkpoint.pt','reference':prefix+'-dev/report.json','passes':1})
    return models
