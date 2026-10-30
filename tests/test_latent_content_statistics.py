import copy
import pytest
from compute_trainium.latent_content_intervention import MODES
from scripts.readout_latent_content import arithmetic_statistics


def fixture():
    out={m:[] for m in MODES}
    for index,b in enumerate(('2','2','3','3','4-6','4-6')):
        for k,m in enumerate(MODES):
            out[m].append(dict(index=index,group=str(index),stratum=b,targets=2,
                answer_nll=1+index/10+k*.1,first_nll=2+index/10+k*.2,exact_prefix_correct=0.))
    return out


def test_sign_counts_and_paired_bin_weighting():
    result=arithmetic_statistics(fixture(),seed=7,replicates=100)
    assert result['cases']==6
    assert result['bins']=={'2':2,'3':2,'4-6':2}
    assert result['means']['authentic']['answer_nll']['mean']==pytest.approx(1.25)
    for k,m in enumerate(MODES[1:],1):
        assert result['removal_penalties'][m]['answer_nll']['mean']==pytest.approx(k*.1)
        assert result['removal_penalties'][m]['first_nll']['conditional_95_interval']==pytest.approx([k*.2,k*.2])


@pytest.mark.parametrize('problem', ['missing','duplicate','group','target','nan'])
def test_rejects_unpaired_or_invalid_metrics(problem):
    rows=copy.deepcopy(fixture())
    if problem=='missing': rows['constant'].pop()
    elif problem=='duplicate': rows['lagged'].append(rows['lagged'][0])
    elif problem=='group': rows['current_only'][0]['group']='changed'
    elif problem=='target': rows['current_only'][0]['targets']=3
    else: rows['constant'][0]['answer_nll']=float('nan')
    with pytest.raises(AssertionError): arithmetic_statistics(rows,seed=7,replicates=100)
