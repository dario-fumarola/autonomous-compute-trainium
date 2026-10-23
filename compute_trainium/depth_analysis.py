"""Frozen seed contrasts and simultaneous intervals for the depth experiment."""
import math
import statistics

from .confirmation import paired_seed_interval

SEEDS = [103, 211, 307]


def family_seed_interval(gains, comparisons=3):
    result=paired_seed_interval(gains)
    # Exact inverse CDF of Student t with df=2, at 1-alpha/(2*m).
    u=1-.05/comparisons
    critical=math.sqrt(2)*u/math.sqrt(1-u*u)
    half=critical*statistics.stdev(gains.values())/math.sqrt(3)
    mean=result['mean_gain_nats'];lo,hi=mean-half,mean+half
    result.update(family_comparisons=comparisons,bonferroni_family_interval_95=[lo,hi],
        family_decision=('USEFUL_GAIN_SUPPORTED' if lo>.01 else
                         ('USEFUL_GAIN_RULED_OUT_AT_THIS_EXPOSURE' if hi<.01 else 'INCONCLUSIVE')))
    return result


def declared_contrasts(rows):
    lookup={(r['arm'],r['seed'],r['endpoint'],r['depth']):r['metrics']['weighted_nll'] for r in rows}
    def gains(a,b):
        return {str(s):lookup[a[0],s,a[1],a[2]]-lookup[b[0],s,b[1],b[2]] for s in SEEDS}
    return {
        'mechanism_mixture_depth1_to_depth3':family_seed_interval(gains(('mixture','1B',1),('mixture','1B',3))),
        'architecture_fixed2_to_mixture2':family_seed_interval(gains(('fixed','1B',2),('mixture','1B',2))),
        'compute_ordinary_matched_to_mixture2':family_seed_interval(gains(('ordinary','matched',1),('mixture','1B',2)))
    }
