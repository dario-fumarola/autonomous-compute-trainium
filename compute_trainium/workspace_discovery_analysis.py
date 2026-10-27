"""Conditional source-group uncertainty;never training-seed confirmation."""
import json
import numpy as np


def clustered_interval(values,groups,domains,*,draws=5000,seed=11261):
    values=np.asarray(values,dtype=np.float64);domains=np.asarray(domains,dtype=np.int64)
    if values.ndim!=1 or len(values)!=len(groups) or len(values)!=len(domains) or not np.isfinite(values).all():raise ValueError('Complete finite paired observations required')
    unique=sorted(set(groups));lookup={g:i for i,g in enumerate(unique)}
    counts=np.zeros((len(unique),3));sums=np.zeros_like(counts)
    for value,group,domain in zip(values,groups,domains,strict=True):
        if domain not in [0,1,2]:raise ValueError('Declared three-domain panel required')
        counts[lookup[group],domain]+=1;sums[lookup[group],domain]+=value
    if (counts.sum(0)==0).any():raise ValueError('No empty evaluation domain')
    weights=np.asarray([.6,.2,.2]);point=float((sums.sum(0)/counts.sum(0))@weights)
    rng=np.random.default_rng(seed);samples=[];rejected=0
    while len(samples)<draws:
        boot=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=draws-len(samples))
        denominator=boot@counts;valid=(denominator>0).all(1);rejected+=int((~valid).sum())
        samples.extend((((boot[valid]@sums)/denominator[valid])@weights).tolist())
        if rejected>100*draws:raise RuntimeError('Too few source groups for stable stratified bootstrap')
    return dict(mean=point,conditional_source_group_95_interval=np.quantile(samples,[.025,.975]).tolist(),
        anchors=len(values),source_groups=len(unique),bootstrap_draws=draws,bootstrap_seed=seed,rejected_empty_domain_draws=rejected,
        warning='Conditional on one trained checkpoint pair and reused TUNE sources.Not independent training-seed variation,confirmation or proof of reasoning.')


def analyze(root,plan):
    native=root/'runs'/(plan['identity']+'-evaluation');rows={}
    for arm in ['c0','ordinary','workspace','pause','draft']:
        records=[r for rank in range(64) for r in json.loads((native/f'rank{rank}-{arm}-losses.json').read_text())]
        records=sorted(records,key=lambda r:r['index'])
        if [r['index'] for r in records]!=list(range(plan['evaluation_anchors'])):raise AssertionError('Exact paired evaluation coverage required')
        rows[arm]=records
    contrasts={
        'ordinary_minus_workspace':(('ordinary','primary_nll'),('workspace','primary_nll')),
        'pause_minus_workspace':(('pause','primary_nll'),('workspace','primary_nll')),
        'draft_minus_workspace':(('draft','primary_nll'),('workspace','primary_nll')),
        'ordinary_minus_workspace_first_token':(('ordinary','primary_first_token_nll'),('workspace','primary_first_token_nll')),
        'pause_minus_workspace_first_token':(('pause','primary_first_token_nll'),('workspace','primary_first_token_nll')),
        'draft_minus_workspace_first_token':(('draft','primary_first_token_nll'),('workspace','primary_first_token_nll')),
        'workspace_raw_minus_own':(('workspace','raw_first4_nll'),('workspace','own_nll')),
        'workspace_donor_minus_own':(('workspace','donor_nll'),('workspace','own_nll')),
        'workspace_constant_minus_own':(('workspace','constant_nll'),('workspace','own_nll')),
        'c0_raw_minus_workspace':(('c0','raw_first4_nll'),('workspace','primary_nll')),
        'ordinary_raw32_minus_workspace_raw32':(('ordinary','raw_nll'),('workspace','raw_nll')),
        'c0_raw32_minus_workspace_raw32':(('c0','raw_nll'),('workspace','raw_nll')),
        'c0_raw_minus_privileged_copy_diagnostic':(('c0','raw_first4_nll'),('c0','privileged_copy_diagnostic_only_nll')),
        'workspace_raw_minus_privileged_copy_diagnostic':(('workspace','raw_first4_nll'),('workspace','privileged_copy_diagnostic_only_nll')),
    }
    domains=[r['domain'] for r in rows['ordinary']];results={}
    for label,(left,right) in contrasts.items():
        values=[a[left[1]]-b[right[1]] for a,b in zip(rows[left[0]],rows[right[0]],strict=True)]
        results[label]=clustered_interval(values,plan['evaluation_groups'],domains)
    return dict(status='COMPLETE_WORKSPACE_DISCOVERY_CONDITIONAL_ANALYSIS',contrasts=results,
        favorable_sign='Positive means the second named predictor has lower NLL;privileged-copy contrasts are diagnostics only.',
        terminal_alpha_spent=0,confirmation=False)
