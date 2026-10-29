"""Numerical report fixtures only; these are not research examples."""
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from compute_trainium.training import file_sha256
from scripts.analyze_budget_workspace import main,PAIRS


def fixture(tmp_path):
    new=tmp_path/'runs'/'new-evaluation';old=tmp_path/'runs'/'old-evaluation'
    new.mkdir(parents=True);old.mkdir(parents=True)
    (new/'report.json').write_text(json.dumps({'status':'COMPLETE_PROBABILITY_SCREEN_EVALUATION'}))
    labels={name for pair in PAIRS.values() for name in pair}
    rows=[]
    for label in sorted(labels):
        for i,domain in enumerate(('web','math','code')):
            rows.append(dict(label=label,panel='broad',index=i,id=str(i),group='group-'+str(i),kind='broad',
                domain=domain,cohort=None,loss_sum=20 if label.startswith('cue-') else 21,targets=10,first_loss=2.))
        for i,kind in enumerate(('code_binding','math_conclusion','math_intermediate_exploratory'),3):
            rows.append(dict(label=label,panel='dependency',index=i,id=str(i),group='group-'+str(i),kind=kind,
                domain=None,cohort=None,loss_sum=20 if label.startswith('cue-') else 21,targets=10,first_loss=2.))
    hashes={}
    for rank in range(64):
        name=f'PRIVATE-rank{rank}-scores.json'
        for folder,is_new in ((new,True),(old,False)):
            (folder/name).write_text(json.dumps([r for r in rows if r['label'].startswith('cue-')==is_new] if rank==0 else []))
        hashes[name]=file_sha256(old/name)
    p=dict(identity='new',parent_evaluation_identity='old-evaluation',parent_score_sha256=hashes,
        bootstrap_seed=9514,bootstrap_replicates=10,practical_gain_nats=.01)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(p))
    return SimpleNamespace(root=tmp_path,plan=plan,output=tmp_path/'analysis'),old


def test_frozen_old_scores_join_with_new_scores_without_retraining(tmp_path):
    args,_=fixture(tmp_path);main(args);r=json.loads((args.output/'report.json').read_text())
    assert r['status']=='COMPLETE_BUDGET_CUE_ANALYSIS'
    assert r['comparisons']['ordinary_equal_work']['broad']['gain']==pytest.approx(.1)
    assert r['screening_gates']['ordinary_equal_work']
    assert not r['screening_gates']['second_replay']
    assert r['reused_scores_sha256_verified'] and r['new_fits']==1


def test_changed_old_control_bytes_are_rejected(tmp_path):
    args,old=fixture(tmp_path);(old/'PRIVATE-rank0-scores.json').write_text('[]')
    with pytest.raises(AssertionError,match='Parent evaluation changed'):main(args)
