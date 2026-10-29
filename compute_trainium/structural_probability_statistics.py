"""Paired source-group diagnostics, with signed absolute effects retained."""
import numpy as np

LABELS=('workspace','ordinary_data','ordinary_work')

def structural_statistics(rows,replicates=2000,seed=9522,minimum_gain=.03):
    cells={}
    for row in rows:
        key=(row['label'],row['condition'],row['kind'])
        cell=cells.setdefault(key,{})
        if row['group'] in cell:raise AssertionError('Exactly one target of each kind per group')
        if row['targets']!=1:raise AssertionError('Single original source token per case')
        cell[row['group']]=float(row['loss_sum'])
    wanted={(label,condition,'structural_'+kind) for label in LABELS for condition in ('full','last32') for kind in ('long','short')}
    if set(cells)!=wanted:raise AssertionError('Complete fixed model/context/case grid required')
    groups=sorted(next(iter(cells.values())))
    if not groups or any(set(c)!=set(groups) for c in cells.values()):raise AssertionError('Exact paired source groups required')
    values={k:np.array([c[g] for g in groups],dtype=np.float64) for k,c in cells.items()}
    if any(not np.isfinite(v).all() for v in values.values()):raise AssertionError('Finite losses required')
    rng=np.random.default_rng(seed);indices=rng.integers(len(groups),size=(replicates,len(groups)))
    def estimate(v):
        lo,hi=np.quantile(v[indices].mean(1),[.025,.975]);return dict(gain=float(v.mean()),conditional_95_interval=[float(lo),float(hi)],source_groups=len(groups))
    v=lambda model,condition,kind:values[(model,condition,'structural_'+kind)]
    means={m:{c:{k:float(v(m,c,k).mean()) for k in ('long','short')} for c in ('full','last32')} for m in LABELS}
    contrasts={}
    for control in ('ordinary_data','ordinary_work'):
        long=v(control,'full','long')-v('workspace','full','long');short=v(control,'full','short')-v('workspace','full','short')
        contrasts['long_vs_'+control]=estimate(long);contrasts['short_vs_'+control]=estimate(short)
        contrasts['long_minus_short_vs_'+control]=estimate(long-short)
    for model in LABELS:
        for kind in ('long','short'):contrasts['full_context_utility_'+model+'_'+kind]=estimate(v(model,'last32',kind)-v(model,'full',kind))
    for control in ('ordinary_data','ordinary_work'):
        contrasts['extra_context_utility_vs_'+control]=estimate((v('workspace','last32','long')-v('workspace','full','long'))-(v(control,'last32','long')-v(control,'full','long')))
    passes=lambda key:contrasts[key]['gain']>=minimum_gain and contrasts[key]['conditional_95_interval'][0]>0
    gates={k:passes('long_vs_'+k) for k in ('ordinary_data','ordinary_work')}
    return dict(means=means,comparisons=contrasts,screening_gates=gates,source_groups=len(groups),bootstrap_replicates=replicates,bootstrap_seed=seed,practical_gain_nats=minimum_gain,
      scope='One trained seed;new source-separated DEVELOPMENT cases. Conditional source intervals do not estimate training-seed variation. Positive long-minus-short interaction alone cannot pass an absolute gain gate. Last32 truncation removes the long opener and keeps the short opener;it is an input/context intervention,not proof of an internal reasoning procedure. No broad-efficiency,iteration or allocation claim follows.')


def test_statistics():
    # Numerical software fixtures,not source/evaluation data.
    rows=[]
    for g in range(8):
        for m,base in [('workspace',1.1),('ordinary_data',1.0),('ordinary_work',.9)]:
            for c in ('full','last32'):
                for kind in ('long','short'):
                    rows.append(dict(label=m,condition=c,kind='structural_'+kind,group=str(g),targets=1,loss_sum=base+(.2 if c=='last32' and kind=='long' else 0)))
    r=structural_statistics(rows,100);assert not any(r['screening_gates'].values());assert abs(r['comparisons']['long_vs_ordinary_data']['gain']+.1)<1e-12
    assert abs(r['comparisons']['full_context_utility_workspace_long']['gain']-.2)<1e-12
    assert abs(r['comparisons']['long_minus_short_vs_ordinary_data']['gain'])<1e-12
    try:structural_statistics(rows+rows[:1],100)
    except AssertionError:pass
    else:raise AssertionError('Duplicate accepted')
    try:structural_statistics(rows[:-1],100)
    except AssertionError:pass
    else:raise AssertionError('Missing control accepted')
    return 5
if __name__=='__main__':print({'software_fixture_checks':test_statistics()})
