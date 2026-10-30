"""Paired source contrasts for the fixed architecture-by-exposure experiment."""
import math
import numpy as np

CELLS=('original-ordinary','original-latent','enriched-ordinary','enriched-latent')
BINS=('2','3','4-6')
CONTRASTS={
    'ordinary_exposure':{'original-ordinary':1,'enriched-ordinary':-1},
    'latent_exposure':{'original-latent':1,'enriched-latent':-1},
    'original_architecture':{'original-ordinary':1,'original-latent':-1},
    'enriched_architecture':{'enriched-ordinary':1,'enriched-latent':-1},
    'interaction':{'original-latent':1,'enriched-latent':-1,'original-ordinary':-1,'enriched-ordinary':1},
}


def arithmetic_statistics(rows,replicates=2000,seed=40532):
    cells={k:{} for k in CELLS}
    for r in rows:
        if r['cell'] not in cells or r['index'] in cells[r['cell']]:
            raise AssertionError('Exactly one score per source and factorial cell')
        if r['stratum'] not in BINS or not all(math.isfinite(r[k]) for k in
                ('answer_nll','first_nll','exact_prefix_correct')):
            raise AssertionError('Finite registered metrics and strata required')
        if r['targets']<1 or r['exact_prefix_correct'] not in (0.,1.):
            raise AssertionError('Nonempty answers and binary prefix correctness')
        cells[r['cell']][r['index']]=r
    ref=cells[CELLS[0]];ids=set(ref)
    if not ids or len({r['group'] for r in ref.values()})!=len(ids):
        raise AssertionError('One independent source group per case required')
    for cell in cells.values():
        if set(cell)!=ids:raise AssertionError('Complete four-cell panel required')
        for i,r in cell.items():
            if any(r[k]!=ref[i][k] for k in ('group','stratum','targets','seed')):
                raise AssertionError('Paired source geometry or initialization changed')
    indices={b:sorted(i for i in ids if ref[i]['stratum']==b) for b in BINS}
    if any(not v for v in indices.values()):raise AssertionError('Every operation bin required')
    rng=np.random.default_rng(seed)
    draws={b:rng.integers(len(v),size=(replicates,len(v))) for b,v in indices.items()}
    def values(key,field):return {b:np.array([cells[key][i][field] for i in indices[b]]) for b in BINS}
    def estimate(v):
        boots=sum(v[b][draws[b]].mean(1) for b in BINS)/len(BINS)
        return dict(mean=float(sum(v[b].mean() for b in BINS)/len(BINS)),
            conditional_95_interval=np.quantile(boots,[.025,.975]).tolist(),
            bins={b:float(v[b].mean()) for b in BINS})
    means={k:{f:estimate(values(k,f)) for f in ('answer_nll','first_nll','exact_prefix_correct')} for k in CELLS}
    contrasts={name:{f:estimate({b:sum(w*values(k,f)[b] for k,w in weights.items()) for b in BINS})
        for f in ('answer_nll','first_nll')} for name,weights in CONTRASTS.items()}
    return dict(cases=len(ids),bins={b:len(v) for b,v in indices.items()},means=means,contrasts=contrasts,
        correct_counts={k:sum(int(v['exact_prefix_correct']) for v in cells[k].values()) for k in CELLS},
        trained_seeds=1,bootstrap_seed=seed,bootstrap_replicates=replicates,
        scope='Source-paired development intervals; not training-seed uncertainty or a compute-efficiency claim.')


def mechanism_gate(arithmetic,broad_gain):
    c=arithmetic['contrasts'];a=c['enriched_architecture']
    return (all(a[f]['mean']>=.03 and a[f]['conditional_95_interval'][0]>0
                for f in ('answer_nll','first_nll'))
        and c['latent_exposure']['first_nll']['mean']>=.03
        and c['interaction']['first_nll']['mean']>0 and broad_gain>=-.005)
