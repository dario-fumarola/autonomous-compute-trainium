"""Separate trained-one-pass contrasts on the unchanged authentic-code panel."""
from .metrics import DocumentScore,summarize,paired_group_interval

BAND_WEIGHTS={'129-256':1/3,'257-512':1/3,'513-900':1/3}


def compare_trained_control(control,candidate,*,draws=5000,seed=1381):
    a={r['id']:r for r in control};b={r['id']:r for r in candidate}
    if len(a)!=len(control) or len(b)!=len(candidate) or a.keys()!=b.keys():
        raise ValueError('Identical unique identifier targets required')
    full_a=[];full_b=[];short_a=[];short_b=[];interaction_a=[];interaction_b=[]
    for key,row in a.items():
        other=b[key]
        if any(row[k]!=other[k] for k in ['group','band','real_targets']):raise ValueError('Control target/source-group pairing differs')
        def score(value):return DocumentScore(row['band'],key,row['group'],value,row['real_targets'])
        af,ash=row['nll_sum']['full'],row['nll_sum']['short']
        bf,bsh=other['nll_sum']['full2'],other['nll_sum']['short2']
        full_a.append(score(af));full_b.append(score(bf));short_a.append(score(ash));short_b.append(score(bsh))
        # (B_full-C_full)-(B_short-C_short),with each original target
        # counted once in the denominator of this explicit linear contrast.
        interaction_a.append(score(af+bsh));interaction_b.append(score(bf+ash))
    interval=lambda left,right:paired_group_interval(left,right,draws=draws,seed=seed,weights=BAND_WEIGHTS)
    return {'trained_onepass_full_nll':summarize(full_a,BAND_WEIGHTS)['weighted_nll'],
        'candidate_two_pass_full_nll':summarize(full_b,BAND_WEIGHTS)['weighted_nll'],
        'trained_onepass_short_nll':summarize(short_a,BAND_WEIGHTS)['weighted_nll'],
        'candidate_two_pass_short_nll':summarize(short_b,BAND_WEIGHTS)['weighted_nll'],
        'candidate_gain_full':interval(full_a,full_b),'candidate_gain_short':interval(short_a,short_b),
        'cross_model_context_interaction':interval(interaction_a,interaction_b),
        'identifier_targets':len(a),'real_target_tokens':sum(r['real_targets'] for r in control),
        'aggregation':'Original equal one-third distance-band weights;tokens pooled within each band.',
        'scope':'Exploratory separately trained control added after candidate diagnostics. Consumed fixed panel and one fitted seed;source intervals do not establish training-seed significance. No changed targets,selected band,unsupported control exit or terminal claim.'}
