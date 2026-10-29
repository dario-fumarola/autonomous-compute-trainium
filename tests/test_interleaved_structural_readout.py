import pytest
from scripts.readout_interleaved_structural import statistics,LABELS


def rows():
    return [dict(label=m,condition=c,kind='structural_'+k,group=str(g),targets=1,
                 loss_sum=1+(.1 if m=='latent' else 0)+(.2 if c=='last32' and k=='long' else 0))
            for g in range(8) for m in LABELS for c in ('full','last32') for k in ('long','short')]


def test_adverse_absolute_results_cannot_be_rescued_by_context_dependence():
    r=statistics(rows(),100)
    assert not any(r['absolute_long_target_gates'].values())
    assert r['comparisons']['long_vs_ordinary']['gain']==pytest.approx(-.1)
    assert r['comparisons']['full_context_utility_latent_long']['gain']==pytest.approx(.2)
    assert r['comparisons']['extra_context_utility_vs_pause']['gain']==pytest.approx(0)


@pytest.mark.parametrize('change',['duplicate','missing','nan'])
def test_incomplete_or_invalid_pairs_rejected(change):
    x=rows()
    if change=='duplicate':x.append(x[0])
    elif change=='missing':x.pop()
    else:x[0]['loss_sum']=float('nan')
    with pytest.raises(AssertionError):statistics(x,100)
