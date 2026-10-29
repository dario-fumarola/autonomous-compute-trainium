import copy
import pytest
from scripts.readout_interleaved_bracket_type import typed_statistics


def fixture():
    rows=[]
    for label in ('latent','ordinary','pause'):
        for condition in ('full','last32'):
            for kind in ('long','short'):
                for closer,n in [(')',4),(']',2),('}',2)]:
                    for i in range(n):
                        value=1.+({'latent':0.,'ordinary':.3,'pause':.2}[label] if kind=='long' else 0.)
                        if condition=='last32':value+=.5
                        rows.append(dict(label=label,condition=condition,kind='structural_'+kind,group=closer+str(i),closer=closer,type_nll=value,type_correct=1))
    return rows


def test_type_statistic_keeps_paired_macro_classes_and_absolute_gate():
    r=typed_statistics(fixture(),replicates=20)
    assert r['groups_by_type']=={')':4,']':2,'}':2}
    assert r['contrasts']['long_vs_ordinary']['mean']==pytest.approx(.3)
    assert r['contrasts']['long_vs_pause']['mean']==pytest.approx(.2)
    assert all(r['useful_long_type_gates'].values())
    assert r['contrasts']['extra_context_utility_vs_pause']['mean']==pytest.approx(0)


def test_type_statistic_rejects_missing_and_mismatched_pairs():
    rows=fixture()
    with pytest.raises(AssertionError):typed_statistics(rows[:-1],replicates=20)
    bad=copy.deepcopy(rows);bad[-1]['closer']=')'
    with pytest.raises(AssertionError):typed_statistics(bad,replicates=20)


def test_unequal_class_counts_do_not_change_macro_weight():
    rows=fixture()
    for r in rows:
        if r['label']=='ordinary' and r['condition']=='full' and r['kind']=='structural_long':
            r['type_nll']=1.+{')':1.,']':0.,'}':0.}[r['closer']]
    result=typed_statistics(rows,replicates=20)
    assert result['contrasts']['long_vs_ordinary']['mean']==pytest.approx(1/3)
