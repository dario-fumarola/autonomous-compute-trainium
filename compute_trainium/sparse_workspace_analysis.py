"""Prespecified paired source-group contrasts;single-seed development only."""
import json
from .workspace_discovery_analysis import clustered_interval


def analyze_sparse(root,plan):
    native=root/'runs'/(plan['identity']+'-evaluation');rows={}
    for arm in ['c0','ordinary','latent','constant']:
        values=[v for rank in range(64) for v in json.loads((native/f'rank{rank}-{arm}-losses.json').read_text())]
        values=sorted(values,key=lambda r:r['index'])
        if [v['index'] for v in values]!=list(range(plan['evaluation_anchors'])):raise AssertionError('Complete paired panel required')
        rows[arm]=values
    contrasts={
        'ordinary_minus_latent_policy32':(('ordinary','policy32_nll'),('latent','policy32_nll')),
        'constant_minus_latent_policy32':(('constant','policy32_nll'),('latent','policy32_nll')),
        'ordinary_minus_latent_anchor4':(('ordinary','raw_anchor_nll'),('latent','workspace_anchor_nll')),
        'constant_minus_latent_anchor4':(('constant','workspace_anchor_nll'),('latent','workspace_anchor_nll')),
        'latent_absolute_anchor_utility':(('latent','raw_anchor_nll'),('latent','workspace_anchor_nll')),
        'initial_c0_absolute_anchor_utility':(('c0','raw_anchor_nll'),('c0','workspace_anchor_nll')),
        'ordinary_minus_latent_raw32':(('ordinary','raw32_nll'),('latent','raw32_nll')),
        'c0_minus_latent_policy32':(('c0','raw32_nll'),('latent','policy32_nll')),
        'latent_constant_input_minus_generated_state':(('latent','constant_input_anchor_nll'),('latent','workspace_anchor_nll')),
        'ordinary_minus_latent_first_target':(('ordinary','raw_first_token_nll'),('latent','workspace_first_token_nll')),
    }
    domains=[v['domain'] for v in rows['ordinary']];result={}
    for key,(left,right) in contrasts.items():
        values=[a[left[1]]-b[right[1]] for a,b in zip(rows[left[0]],rows[right[0]],strict=True)]
        result[key]=clustered_interval(values,plan['evaluation_groups'],domains)
    scores={arm:{field:sum(w*sum(v[field] for v in values if v['domain']==d)/sum(v['domain']==d for v in values) for d,w in enumerate([.6,.2,.2])) for field in values[0] if field.endswith('_nll')} for arm,values in rows.items()}
    published=json.loads((native/'report.json').read_text())['scores']
    discrepancy=max(abs(value-published[arm][field]) for arm,fields in scores.items() for field,value in fields.items())
    if discrepancy>1e-10:raise AssertionError('Independent row aggregation disagrees')
    return dict(status='COMPLETE_SPARSE_CONDITIONAL_ANALYSIS',contrasts=result,independent_scores=scores,aggregate_max_abs_error=discrepancy,domains={str(d):domains.count(d) for d in range(3)},terminal_alpha_spent=0,confirmation=False,warning='One trained seed and reused TUNE groups;conditional source-group intervals omit training-seed variation.No allocation,reasoning or random-init conclusion.')
