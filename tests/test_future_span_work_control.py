import copy
import pytest

from compute_trainium.future_span_work_control import (
    admission_evidence, make_cases, measured_control_budget, validate_control_score,
)
from compute_trainium.ntp_secondary import ntp_point_contrasts, discovery_decision


def fixture():
    # Analysis metadata only;no model inputs or generated research examples.
    cases, rows, gates = {}, [], []
    for seed in [5231, 7283]:
        for cell in ['one-ntp', 'one-future', 'recurrent-ntp', 'recurrent-future']:
            key = f'{cell}-{seed}'
            case = dict(cell=cell, seed=seed, arm='one' if cell.startswith('one') else 'recurrent',
                total_updates=40, end_targets=2000, training_parameters=10,
                cooldown_start_update=20, cooldown_start_targets=1000, warmup_targets=100,
                peak_lr=.032, state_seed=seed+2009)
            cases[key] = case
            for depth in [1] if case['arm'] == 'one' else [1, 2, 3]:
                loss = {'one-ntp': .50, 'one-future': .49}.get(cell, .48 if depth == 1 else .45)
                rows.append(dict(cell=cell, seed=seed, case_id=key, depth=depth, training_update=40,
                    training_real_targets=2000, training_parameters=10, deployed_parameters=8,
                    checkpoint_sha256=key, metrics={'weighted_nll': loss}, readout='cooled1b'))
            gates.append({'case_id': key, 'independent_cpu': dict(
                status='PASS_INDEPENDENT_MATURE_FUTURE_SPAN_GRADIENT', training_update=40,
                checkpoint_sha256=key)})
    points = {s: ntp_point_contrasts([r for r in rows if r['seed'] == s]) for s in [5231, 7283]}
    parent = dict(status='COMPLETE_BOUNDED_FUTURE_SPAN_OBJECTIVE_PILOT', scores=rows,
                  independent_gradient_gates=gates, decision={'decision': 'unchanged'})
    secondary = dict(status='COMPLETE_NTP_SECONDARY_DISCOVERY_READOUT',
        points_by_pilot_seed=points, secondary_decision=discovery_decision(points))
    return parent, {'cases': cases}, secondary


def test_admission_requires_both_seeds_all_paths_and_exact_numerical_gates():
    parent, plan, secondary = fixture()
    assert admission_evidence(parent, plan, secondary)['eligible']
    for mutation in ['missing_path', 'wrong_gate', 'edited_secondary']:
        p, m, s = copy.deepcopy((parent, plan, secondary))
        if mutation == 'missing_path': p['scores'].pop()
        elif mutation == 'wrong_gate': p['independent_gradient_gates'][0]['independent_cpu']['checkpoint_sha256'] = 'wrong'
        else: s['points_by_pilot_seed'][5231]['over_one_ntp'] = 99
        with pytest.raises(ValueError): admission_evidence(p, m, s)
    for r in parent['scores']:
        if r['seed'] == 7283 and r['cell'] == 'recurrent-ntp' and r['depth'] == 2:
            r['metrics']['weighted_nll'] = .499
    points = {s: ntp_point_contrasts([r for r in parent['scores'] if r['seed'] == s]) for s in [5231, 7283]}
    secondary.update(points_by_pilot_seed=points, secondary_decision=discovery_decision(points))
    assert not admission_evidence(parent, plan, secondary)['eligible']


def test_cases_keep_the_shared_history_and_count_preflight_as_science():
    _, plan, _ = fixture()
    geometry = dict(origin_update=20, total_updates=60, cooldown_start_update=30,
        origin_source={'real_targets': 1000}, final_source={'real_targets': 3000, 'source_cursor': 120},
        cooldown_source={'real_targets': 1500}, matched_work={'executed_matmuls': 9000})
    result = make_cases(plan, geometry, list(range(121)))
    assert len(result) == 4
    for case in result.values():
        assert case['preflight_end_update'] == 22
        assert case['segment_ends'] == [22, 60] and case['gradient_updates'] == [22, 60]
        assert case['parent']['update'] == 20 and case['end_anchors'] == 120
        original = plan['cases'][case['source_parent_case_id']]
        assert case['peak_lr'] == original['peak_lr'] and case['warmup_targets'] == original['warmup_targets']
    bad = copy.deepcopy(geometry);bad['cooldown_start_update'] = 21
    with pytest.raises(ValueError): make_cases(plan, bad, list(range(121)))


def test_budget_counts_complete_cpu_waves_and_rejects_short_measurements():
    hardware = {'status': 'PASS_COMPLETE_FUTURE_SPAN_TRAINING_QUALIFICATION',
        'records': [{'case': {'identity': c}, 'native': {'warm': dict(warm_updates=100,
            warm_seconds=100., wall_seconds=200.)}} for c in ['one-ntp', 'one-future', 'recurrent-ntp', 'recurrent-future']]}
    result = measured_control_budget(hardware)
    assert result['fits_registered_cap'] and result['cpu_and_transport_seconds_allowance'] == 2460
    bad = copy.deepcopy(hardware);bad['records'][0]['native']['warm']['warm_updates'] = 12
    with pytest.raises(ValueError): measured_control_budget(bad)


def test_readout_rejects_wrong_checkpoint_work_and_consumed_panel_identity():
    case = dict(seed=5231, cell='one-ntp', total_updates=60, end_targets=3000,
        end_anchors=100, training_dense_matrix_ops=9000, training_parameters=10)
    plan = {'evaluation': {'corpus_sha256': 'original-panel'}}
    row = dict(case_id='control', seed=5231, cell='one-ntp', depth=1,
        training_update=60, training_real_targets=3000, training_valid_anchors=100,
        estimated_training_dense_matrix_ops=9000, training_parameters=10,
        checkpoint_sha256='endpoint', source_corpus_sha256='original-panel',
        cohort='curve', split='dev', head_removed=True, full_state_readback_exact=True,
        model_unchanged=True, test_accessed=False)
    validate_control_score(row, case, 'control', 'endpoint', plan)
    for key, value in [('checkpoint_sha256', 'other'), ('estimated_training_dense_matrix_ops', 8000),
                       ('training_real_targets', 2000), ('source_corpus_sha256', 'new-panel'),
                       ('head_removed', False), ('training_parameters', 11)]:
        with pytest.raises(ValueError):
            validate_control_score({**row, key: value}, case, 'control', 'endpoint', plan)


def test_expired_parent_window_cannot_admit_even_when_results_are_available(tmp_path, monkeypatch):
    import json
    from scripts import run_future_span_work_controls as runner
    (tmp_path/'future-span-main-v2').mkdir()
    (tmp_path/'future-span-main-v2/plan.json').write_text(json.dumps({'cases': {}}))
    monkeypatch.setattr(runner, 'source_integrity', lambda _: None)
    with pytest.raises(TimeoutError, match='parent admission window'):
        runner.wait_and_register(tmp_path, {'source_sha256': {}, 'wait_deadline_unix': 0})
