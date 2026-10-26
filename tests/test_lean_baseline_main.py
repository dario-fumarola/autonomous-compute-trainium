import pytest
from scripts.run_lean_baseline_main import selected_rates,identity,project_main_seconds


def test_stronger_controls_require_complete_qualified_rate_selection():
    plans={r:{'cases':{r+'-lr'+str(i):{'peak_lr':lr} for i,lr in enumerate([.008,.016,.032])}} for r in ['mixing','ordinary8']}
    tune=dict(status='COMPLETE_QUALIFIED_LEAN_BASELINE_TUNING',selected={},selected_rates={},scores={},selected_final_gates={})
    for role,p in plans.items():
        for i,key in enumerate(p['cases']):tune['scores'][key]={'metrics':{'weighted_nll':3-i*.01}}
        winner=role+'-lr2';tune['selected'][role]=winner;tune['selected_rates'][role]=.032
        tune['selected_final_gates'][role]={'independent_cpu':dict(status='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT',case_id=winner)}
    assert selected_rates(tune,plans)==dict(one=.032,mixing=.032,ordinary8=.032)
    tune['selected_final_gates']['mixing']['independent_cpu']['status']='FAIL'
    with pytest.raises(AssertionError):selected_rates(tune,plans)


def test_each_main_seed_has_a_measured_complete_finite_cap():
    profiles={r:{'matched_work':{'updates':u}} for r,u in [('one',4706),('mixing',4674),('ordinary8',5275)]}
    qualified={'records':{r:{'warm':{'warm_seconds':s}} for r,s in [('one',119.24426625296474),('mixing',116.16462083300576),('ordinary8',94.0157756889239)]}}
    result=project_main_seconds(profiles,qualified)
    assert 15000<result['complete_seconds']<18000
    assert identity(17293)=='lean-baseline-main-s17293-v1'
    with pytest.raises(ValueError):identity(999)


def test_recurrence_advantage_is_positive_when_cheaper_baseline_loss_is_higher(tmp_path):
    import json
    from compute_trainium.training import file_sha256
    from scripts.run_lean_baseline_main import paired_comparison
    reference=tmp_path/'reference.json';baseline=tmp_path/'baseline';baseline.mkdir()
    rows=[dict(domain=d,id=d+'-'+g,group=g,nll_sum=20.,real_targets=10)
          for g in ['group-a','group-b'] for d in ['web','math','code']]
    reference.write_text(json.dumps({'records':rows}))
    (baseline/'document-scores.json').write_text(json.dumps({'records':[{**r,'nll_sum':21.} for r in rows]}))
    report=paired_comparison(tmp_path,dict(document_scores='reference.json',document_scores_sha256=file_sha256(reference)),baseline)
    assert report['gain_nats']==pytest.approx(.1)
