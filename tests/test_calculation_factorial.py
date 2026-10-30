import copy
import pytest
from compute_trainium.calculation_factorial import CELLS,arithmetic_statistics,mechanism_gate


def rows():
    # Software-only loss fixtures, never natural research examples.
    loss=dict(zip(CELLS,(2.,1.9,1.8,1.6)))
    return [dict(cell=c,index=i,group=str(i),stratum=b,targets=2,seed=1,
        answer_nll=loss[c],first_nll=loss[c],exact_prefix_correct=0.)
        for c in CELLS for i,b in enumerate(('2','3','4-6'))]


def test_interaction_sign_and_macro_pairing():
    a=arithmetic_statistics(rows(),replicates=20)
    assert a['contrasts']['ordinary_exposure']['first_nll']['mean']==pytest.approx(.2)
    assert a['contrasts']['latent_exposure']['first_nll']['mean']==pytest.approx(.3)
    assert a['contrasts']['interaction']['first_nll']['mean']==pytest.approx(.1)
    assert mechanism_gate(a,0.)


@pytest.mark.parametrize('change',['missing','duplicate','group','targets','nan','seed'])
def test_rejects_unpaired_or_invalid_rows(change):
    v=rows()
    if change=='missing':v.pop()
    elif change=='duplicate':v.append(copy.deepcopy(v[-1]))
    elif change=='nan':v[-1]['first_nll']=float('nan')
    else:v[-1][change]='changed' if change=='group' else 3
    with pytest.raises(AssertionError):arithmetic_statistics(v,replicates=20)


def test_later_token_or_general_data_gain_does_not_pass():
    a=arithmetic_statistics(rows(),replicates=20)
    a['contrasts']['enriched_architecture']['first_nll']['mean']=.001
    assert not mechanism_gate(a,0.)
    a=arithmetic_statistics(rows(),replicates=20)
    a['contrasts']['interaction']['first_nll']['mean']=0.
    assert not mechanism_gate(a,0.)
    a=arithmetic_statistics(rows(),replicates=20)
    assert not mechanism_gate(a,-.006)
