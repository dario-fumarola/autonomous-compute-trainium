"""Report arithmetic fixtures; never research examples."""
import json
from types import SimpleNamespace
import pytest
from compute_trainium.training import file_sha256
from scripts.analyze_untied_probability import main,interaction_rows,PAIRS


def fixture(tmp_path,values=None):
    new=tmp_path/'runs/new-evaluation';old=tmp_path/'runs/old-evaluation';new.mkdir(parents=True);old.mkdir(parents=True)
    (new/'report.json').write_text(json.dumps({'status':'COMPLETE_PROBABILITY_SCREEN_EVALUATION'}))
    vals={'untied-one-k1':2.3,'untied-ordinary-data-k0':2.6,'untied-ordinary-work-k0':2.4,
      'fixed-final-k1':2.,'ordinary_data-k0':2.2,'ordinary_work-k0':2.1,'untied-one-permuted-k1':3.}
    vals.update(values or {});rows=[]
    for label,loss in vals.items():
        for i,(domain,kind) in enumerate([('web','broad'),('math','broad'),('code','broad'),(None,'code_binding'),(None,'math_conclusion'),(None,'math_intermediate_exploratory')]):
            rows.append(dict(label=label,id=str(i),group='g'+str(i),domain=domain,kind=kind,panel='broad' if kind=='broad' else 'dependency',targets=10,loss_sum=10*loss,first_loss=loss))
    hashes={}
    for rank in range(64):
        n=f'PRIVATE-rank{rank}-scores.json'
        for folder,is_new in ((new,True),(old,False)):(folder/n).write_text(json.dumps([r for r in rows if r['label'].startswith('untied-')==is_new] if rank==0 else []))
        hashes[n]=file_sha256(old/n)
    p=tmp_path/'plan.json';p.write_text(json.dumps(dict(identity='new',parent_evaluation_identity='old-evaluation',parent_score_sha256=hashes,bootstrap_seed=5,bootstrap_replicates=20,practical_gain_nats=.01)))
    return SimpleNamespace(root=tmp_path,plan=p,output=tmp_path/'analysis'),old


def test_absolute_controls_and_paired_interaction_have_correct_sign(tmp_path):
    a,_=fixture(tmp_path);main(a);r=json.loads((a.output/'report.json').read_text())
    assert r['comparisons']['ordinary_equal_data']['broad']['gain']==pytest.approx(.3)
    assert r['comparisons']['ordinary_equal_work']['broad']['gain']==pytest.approx(.1)
    assert r['comparisons']['untying_interaction_equal_data']['broad']['gain']==pytest.approx(.1)
    assert r['comparisons']['untying_interaction_equal_work']['broad']['gain']==pytest.approx(0)
    assert all(r['screening_gates'].values()) and r['new_fits']==3


def test_interaction_does_not_promote_an_absolute_loss(tmp_path):
    a,_=fixture(tmp_path,{'ordinary_data-k0':1.8,'untied-ordinary-data-k0':2.29});main(a);r=json.loads((a.output/'report.json').read_text())
    assert r['comparisons']['untying_interaction_equal_data']['broad']['gain']>0
    assert not r['screening_gates']['ordinary_equal_data']


def test_hash_tampering_and_interaction_target_mismatch_fail(tmp_path):
    a,old=fixture(tmp_path);rows=json.loads((old/'PRIVATE-rank0-scores.json').read_text())
    (old/'PRIVATE-rank0-scores.json').write_text('[]')
    with pytest.raises(AssertionError,match='bytes changed'):main(a)
    rs=[dict(label=l,id='a',group='g',targets=10,domain='web',loss_sum=20,first_loss=2) for l in ('untied-ordinary-data-k0','fixed-final-k1','untied-one-k1','ordinary_data-k0')]
    rs[-1]['targets']=11
    with pytest.raises(AssertionError,match='grouping/targets'):interaction_rows(rs,'untied-ordinary-data-k0')
