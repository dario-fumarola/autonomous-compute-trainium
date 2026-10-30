"""Statistical fixtures, never research cases."""
import pytest
from scripts.readout_direct_arithmetic import statistics,SEEDS,ROLES,BINS


def fixture():
    return [dict(index=i,group=str(i),stratum=b,seed=s,role=r,targets=i+1,
                 answer_nll=1.+i+(.1 if r!='latent' else 0),first_nll=2.,exact_prefix_correct=0.)
            for s in SEEDS for r in ROLES for i,b in enumerate(BINS)]


def test_equal_bin_case_weighting_and_pairing():
    r=statistics(fixture(),replicates=10)
    assert r['task_specific_lead']
    assert r['means']['9901']['latent']['answer_nll']['macro']['mean']==2.
    assert r['across_seeds']['ordinary']['mean']==pytest.approx(.1)


def test_missing_or_duplicate_records_rejected():
    rows=fixture()
    with pytest.raises(AssertionError):statistics(rows[:-1],replicates=10)
    with pytest.raises(AssertionError):statistics(rows+[rows[-1]],replicates=10)


def test_one_weak_seed_does_not_pass_development_rule():
    rows=fixture()
    for row in rows:
        if row['seed']==730201 and row['role']=='latent':row['answer_nll']+=.09
    assert not statistics(rows,replicates=10)['task_specific_lead']
