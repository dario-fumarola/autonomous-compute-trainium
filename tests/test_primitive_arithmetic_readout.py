import pytest
from scripts.readout_primitive_arithmetic import statistics, BINS, ROLES


def fixture():
    return [dict(index=i,group=str(i),stratum=b,seed=3,role=r,targets=2,
                 answer_nll=float(i)+(r!='latent'),first_nll=float(i),exact_prefix_correct=float(r=='latent'))
            for i,b in enumerate([BINS[0],BINS[0],*BINS[1:]]) for r in ROLES]


def test_equal_operator_macro_and_paired_differences():
    out = statistics(fixture(),replicates=20)
    assert out['means']['latent']['answer_nll']['mean'] == pytest.approx((.5+2+3+4)/4)
    assert out['contrasts']['ordinary']['answer_nll']['mean'] == pytest.approx(1.)
    assert out['contrasts']['ordinary']['exact_prefix_correct']['mean'] == pytest.approx(1.)


@pytest.mark.parametrize('change', ['missing','target','source','nan','operator'])
def test_incomplete_or_unpaired_scores_rejected(change):
    rows = fixture()
    if change == 'missing': rows.pop()
    elif change == 'target': rows[-1]['targets'] = 3
    elif change == 'source': rows[-1]['group'] = 'different'
    elif change == 'nan': rows[-1]['answer_nll'] = float('nan')
    else: rows[-1]['stratum'] = 'other'
    with pytest.raises(AssertionError): statistics(rows,replicates=20)
