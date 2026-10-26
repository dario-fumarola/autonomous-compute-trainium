"""Numerical admission fixtures,not model research data."""
import pytest

from compute_trainium.late_allocation_main import bounded_deadline,qualified_rate,seed_summary,validate_rng_inventory,final_score_gate


def gate(case,update,sha='fixture-sha'):
    common=dict(case_id=case,training_update=update,checkpoint_sha256=sha,windows=512,model_unchanged=True,optimizer_updates=0)
    return dict(native=dict(common,status='CAPTURED_ZERO_UPDATE_FUTURE_SPAN_GRADIENT',replica_gradients_equal=True),
        independent_cpu=dict(common,status='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT',full_state_readback_exact=True,
            gradient=dict(body=dict(overall=dict(relative_l2=.001),families=dict(body=dict(relative_l2=.001)),passes_two_percent_gate=True),future_head=dict(relative_l2=0.),future_head_named={'weight':dict(relative_l2=0.)},passes_two_percent_gate=True),
            loss_differences=dict(ntp_nll=0.,future_mean=0.)))


def fixture():
    cases={str(lr):dict(peak_lr=lr,end_targets=123,training_dense_matrix_ops=456) for lr in [.008,.016,.032]}
    p=dict(cases=cases,evaluation=dict(corpus_sha256='panel'))
    r=dict(status='COMPLETE_QUALIFIED_LATE_TWO_ALLOCATION_TUNING',scores={},prefix_gates={},completed_fit_reports={},actual_warm={},switch_readback_checks={},scientific_optimizer_updates=1728,selected_case='.032',selected_rate=.032)
    for i,key in enumerate(cases):
        r['scores'][key]=dict(status='COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE',case_id=key,seed=4127,training_update=576,depth=2,cohort='tune',objective_weight=0.,source_corpus_sha256='panel',training_real_targets=123,estimated_training_dense_matrix_ops=456,metrics=dict(weights={'web':.6,'math':.2,'code':.2},weighted_nll=3.-.1*i),checkpoint_sha256='fixture-sha',model_unchanged=True,full_state_readback_exact=True)
        r['prefix_gates'][key]=gate(key,121)
        r['completed_fit_reports'][key]=dict(update=576,complete_state_on_load_exact=True,cpu_rng_on_load_exact=True,checkpoint_records=[dict(all_complete_replica_states_equal=True)],progress=[dict(update=576,lr=0.)])
        r['actual_warm'][key]=dict(one_pass_updates=100,two_pass_updates=100,one_pass_seconds=110.,two_pass_seconds=140.)
        r['switch_readback_checks'][key]=dict(update=432,restored_complete_state=True,restored_cpu_rng=True,unchanged_endpoint_schedule=True)
    r['selected_case']='0.032';r['selected_final_gate']=gate('0.032',576)
    return r,p


def test_complete_grid_selects_rate_but_nominal_complete_cannot_hide_missing_trial():
    r,p=fixture();assert qualified_rate(r,p)==.032
    del r['prefix_gates']['0.008']
    with pytest.raises(ValueError):qualified_rate(r,p)


def test_wrong_checkpoint_and_failed_family_are_rejected_even_with_success_label():
    r,p=fixture();r['selected_final_gate']['independent_cpu']['checkpoint_sha256']='different'
    with pytest.raises(ValueError):qualified_rate(r,p)
    r,p=fixture();r['prefix_gates']['0.016']['independent_cpu']['gradient']['body']['families']['body']['relative_l2']=.021
    with pytest.raises(ValueError):qualified_rate(r,p)


def test_interrupted_cooldown_or_switch_rng_recovery_blocks_admission():
    r,p=fixture();r['completed_fit_reports']['0.008']['progress'][-1]['lr']=.001
    with pytest.raises(ValueError):qualified_rate(r,p)
    r,p=fixture();r['switch_readback_checks']['0.016']['restored_cpu_rng']=False
    with pytest.raises(ValueError):qualified_rate(r,p)


def test_composite_deadline_can_only_shorten_relative_main_cap():
    assert bounded_deadline(100.,500.)==500.
    assert bounded_deadline(100.,20000.)==14500.
    with pytest.raises(ValueError):bounded_deadline(500.,500.)


def test_seed_gate_retains_unfavorable_pair_and_cannot_claim_confirmation():
    r=seed_summary({5231:.02,17293:-.001})
    assert not r['development_point_gate'] and not r['terminal_success']
    assert r['paired_seed_gains']['17293']==-.001
    with pytest.raises(ValueError):seed_summary({5231:.02})


def test_rng_inventory_rejects_missing_or_unrecoverable_rank_state():
    import torch
    states=[torch.Generator().manual_seed(i).get_state() for i in range(32)]
    assert validate_rng_inventory(states)
    with pytest.raises(ValueError):validate_rng_inventory(states[:-1])
    states[5]=torch.zeros(3,dtype=torch.float32)
    with pytest.raises(ValueError):validate_rng_inventory(states)


def test_real_published_gradient_receipt_schema_and_named_head_gate():
    import copy,json
    from pathlib import Path
    from compute_trainium.late_allocation_main import gradient_gate
    path=Path(__file__).parents[1]/'results/fixed_two_small_core_tune_20261025/report.json'
    report=json.loads(path.read_text());key=next(iter(report['prefix_gates']));actual=report['prefix_gates'][key]
    assert gradient_gate(actual,key,121)
    changed=copy.deepcopy(actual)
    changed['independent_cpu']['gradient']['future_head_named']['future_lm_head.weight']['relative_l2']=.021
    with pytest.raises(ValueError):gradient_gate(changed,key,121)


def test_scoring_cannot_substitute_wrong_seed_panel_or_checkpoint():
    case=dict(seed=5231,end_targets=123,training_dense_matrix_ops=456)
    plan=dict(cases={'main':case},evaluation=dict(corpus_sha256='panel'))
    row=dict(status='COMPLETE_INDEPENDENT_FUTURE_SPAN_DEVELOPMENT_SCORE',case_id='main',seed=5231,
        depth=2,cohort='curve',objective_weight=0.,training_update=4400,training_parameters=166571840,
        deployed_parameters=141963840,checkpoint_sha256='checkpoint',source_corpus_sha256='panel',
        training_real_targets=123,estimated_training_dense_matrix_ops=456,
        full_state_readback_exact=True,model_unchanged=True,metrics=dict(weights={'web':.6,'math':.2,'code':.2},weighted_nll=2.5))
    assert final_score_gate(row,plan,'main','checkpoint')
    for field,value in [('seed',17293),('source_corpus_sha256','other'),('checkpoint_sha256','other')]:
        changed=dict(row,**{field:value})
        with pytest.raises(ValueError):final_score_gate(changed,plan,'main','checkpoint')
