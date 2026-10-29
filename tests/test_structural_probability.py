"""Software fixtures for paired source statistics and context interventions."""
import numpy as np
import pytest
from compute_trainium.structural_probability_statistics import structural_statistics
from scripts.evaluate_structural_probability import truncated_arrays


def rows():
    return [dict(label=m,condition=c,kind='structural_'+k,group=str(g),targets=1,loss_sum=b+(.2 if c=='last32' and k=='long' else 0))
            for g in range(8) for m,b in [('workspace',1.1),('ordinary_data',1.),('ordinary_work',.9)]
            for c in ('full','last32') for k in ('long','short')]


def test_absolute_effect_sign_context_and_pairing():
    r=structural_statistics(rows(),100)
    assert r['comparisons']['long_vs_ordinary_data']['gain']==pytest.approx(-.1)
    assert r['comparisons']['full_context_utility_workspace_long']['gain']==pytest.approx(.2)
    assert r['comparisons']['long_minus_short_vs_ordinary_data']['gain']==pytest.approx(0)
    assert not any(r['screening_gates'].values())


def test_positive_interaction_does_not_promote_absolute_negative():
    rs=rows()
    for r in rs:
        if r['label']=='workspace' and r['kind']=='structural_short':r['loss_sum']+=.3
    report=structural_statistics(rs,100)
    assert report['comparisons']['long_minus_short_vs_ordinary_data']['gain']>0
    assert not report['screening_gates']['ordinary_data']


def test_complete_grid_duplicate_and_nonfinite_guards():
    rs=rows()
    with pytest.raises(AssertionError):structural_statistics(rs[:-1],100)
    with pytest.raises(AssertionError):structural_statistics(rs+rs[:1],100)
    rs[0]['loss_sum']=float('nan')
    with pytest.raises(AssertionError):structural_statistics(rs,100)


def test_favorable_absolute_effect_and_deterministic_interval():
    rs=rows()
    for r in rs:
        if r['label']=='workspace':r['loss_sum']-=.25
    one=structural_statistics(rs,100);two=structural_statistics(list(reversed(rs)),100)
    assert one==two and all(one['screening_gates'].values())


def source_fixture():
    x=np.full((2,512),2,dtype=np.uint16);y=x.copy();m=np.zeros((2,512),dtype=np.bool_);meta=[]
    for i,(length,kind,distance) in enumerate([(120,'structural_long',80),(60,'structural_short',5)]):
        x[i,:length]=np.arange(10,10+length);y[i,:length]=np.arange(11,11+length);m[i,length-1]=True
        meta.append(dict(prefix_tokens=length,target_start=length-1,kind=kind,opener_distance=distance))
    return x,y,m,meta


def test_context_restriction_retains_exact_label_and_last_source_tokens():
    x,y,m,meta=source_fixture();xx,yy,mm=truncated_arrays(x,y,m,meta)
    assert np.array_equal(yy[mm],y[m]) and mm.sum()==2
    for i,r in enumerate(meta):
        n=r['prefix_tokens'];assert np.array_equal(xx[i,:32],x[i,n-32:n]);assert np.all(xx[i,32:]==2)
    assert (mm[:,31]).all() and not np.any(mm[:,:31])


def test_invalid_target_and_antecedent_masks_rejected():
    x,y,m,meta=source_fixture();m[0,0]=True
    with pytest.raises(AssertionError):truncated_arrays(x,y,m,meta)
    x,y,m,meta=source_fixture();meta[0]['opener_distance']=20
    with pytest.raises(AssertionError):truncated_arrays(x,y,m,meta)
    x,y,m,meta=source_fixture();meta[1]['opener_distance']=80
    with pytest.raises(AssertionError):truncated_arrays(x,y,m,meta)
