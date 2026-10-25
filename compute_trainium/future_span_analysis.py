"""Declared point contrasts;pilot screening is not terminal significance."""
import math

from .future_span_study_plan import CELLS
from .metrics import DocumentScore,paired_group_interval


def point_contrasts(scores):
    required={(cell,1 if cell.startswith('one') else 2) for cell in CELLS}
    required.add(('recurrent-future',1))
    if not required.issubset(scores) or any(not math.isfinite(scores[k]) for k in required):
        raise ValueError('All four primary cells and the supported one-pass recurrent exit required')
    o,f,r,rf=(scores[(cell,1 if cell.startswith('one') else 2)] for cell in CELLS)
    return {'objective_recurrence_interaction':(f-rf)-(o-r),
        'recurrent_future_over_one_future':f-rf,'recurrent_future_over_recurrent_ntp':r-rf,
        'recurrent_future_depth1_to2':scores[('recurrent-future',1)]-rf,
        'ordinary_future_objective_gain':o-f,'recurrent_future_objective_gain':r-rf,
        'recurrence_without_future_gain':o-r,'recurrence_with_future_gain':f-rf}


def interaction_group_interval(documents,*,draws=2000,seed=5231):
    """Paired linear contrast with one unchanged denominator per source target."""
    lookup={cell:{r.document:r for r in documents[cell]} for cell in CELLS}
    base=lookup['one-ntp']
    if any(set(rows)!=set(base) for rows in lookup.values()):raise ValueError('All four cells need identical unique source documents')
    if any(len(documents[cell])!=len(lookup[cell]) for cell in CELLS):raise ValueError('Repeated source document')
    a=[];b=[]
    for key,row in base.items():
        if any((r[key].domain,r[key].group,r[key].tokens)!=(row.domain,row.group,row.tokens) for r in lookup.values()):
            raise ValueError('Four-cell targets/source groups differ')
        # These are explicit sums of losses,not probabilities of invented
        # models. Both keep the same original target denominator,not twice it.
        a.append(DocumentScore(row.domain,key,row.group,lookup['one-future'][key].nll_sum+lookup['recurrent-ntp'][key].nll_sum,row.tokens))
        b.append(DocumentScore(row.domain,key,row.group,lookup['recurrent-future'][key].nll_sum+row.nll_sum,row.tokens))
    result=paired_group_interval(a,b,draws=draws,seed=seed)
    result['scope']='Conditional source-group bootstrap of the paired four-cell linear interaction;not training-seed or terminal uncertainty.'
    return result


def pilot_decision(per_seed,threshold=.01):
    if set(per_seed)!={5231,7283}:raise ValueError('Both frozen paired pilot seeds required')
    keys=['objective_recurrence_interaction','recurrent_future_over_one_future',
          'recurrent_future_over_recurrent_ntp','recurrent_future_depth1_to2']
    point_pass=all(per_seed[seed][key]>=threshold for seed in per_seed for key in keys)
    return {'both_seeds_clear_all_point_gates':point_pass,
        'decision':'CANDIDATE_FOR_STRONG_BASELINE_AND_INDEPENDENT_CONFIRMATION' if point_pass else 'CLOSE_THIS_BOUNDED_OBJECTIVE_RECIPE',
        'useful_effect_threshold_nats':threshold,'terminal_success':False,'terminal_alpha_spent':0,
        'scope':'Two discovery seeds on consumed DEV. Point gates are screening evidence,not statistical significance. A practical claim also requires the strongest eligible one-pass+future comparator and resource controls;terminal confirmation requires fresh sources/seeds and prospective alpha.'}
