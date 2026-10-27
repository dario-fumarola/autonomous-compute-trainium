import json
import pytest
from compute_trainium.training import file_sha256
from scripts import run_scale_boundary_lower_rates as module


def records():
    rates = {}; rows = {}
    for prefix in ['fixed-two', 'ordinary8']:
        for i, rate in enumerate(module.GRID):
            key = prefix + '-fixture-' + str(i)
            rates[key] = rate
            rows[key] = {'score': {'metrics': {'weighted_nll': 3.}}}
    return rows, rates


def test_lower_check_is_symmetric_and_does_not_expand_upper_boundary():
    assert module.needs_lower_check({'fixed_two': {'rate': .008}, 'ordinary8': {'rate': .016}})
    assert module.needs_lower_check({'fixed_two': {'rate': .016}, 'ordinary8': {'rate': .008}})
    assert not module.needs_lower_check({'fixed_two': {'rate': .064}, 'ordinary8': {'rate': .032}})


def test_selection_requires_all_opportunities_and_a_healthy_role():
    rows, rates = records()
    del rows['ordinary8-fixture-0']
    with pytest.raises(ValueError, match='All six'):module.select(rows, rates)
    rows, rates = records()
    for key in rows:
        if key.startswith('ordinary8'):rows[key]['score']['metrics']['weighted_nll'] = 4.
    with pytest.raises(ValueError, match='No healthy'):module.select(rows, rates)


def test_independent_selection_uses_minimum_score_then_lower_rate():
    rows, rates = records()
    rows['ordinary8-fixture-3']['score']['metrics']['weighted_nll'] = 2.9
    selected = module.select(rows, rates)
    assert selected['fixed_two']['rate'] == .002
    assert selected['fixed_two']['boundary_optimum']
    assert selected['ordinary8']['rate'] == .016
    assert not selected['ordinary8']['boundary_optimum']


def test_registered_trigger_prevents_main_fallback_to_original_selection(tmp_path):
    parent = tmp_path / module.PARENT; parent.mkdir()
    plan = parent / 'plan.json'; plan.write_text('{}')
    result = dict(status='COMPLETE_BOUNDARY_INDEPENDENT_RATE_SELECTION',
                  plan_sha256=file_sha256(plan),
                  selected={'fixed_two': {'rate': .008}, 'ordinary8': {'rate': .008}})
    (parent / 'report.json').write_text(json.dumps(result))
    assert module.selection_source(tmp_path) == (plan, parent / 'report.json')
    scope = tmp_path / module.SCOPE; scope.mkdir()
    (scope / 'scope.json').write_text(json.dumps({'parent_plan_sha256':file_sha256(plan)}))
    with pytest.raises(FileNotFoundError):module.selection_source(tmp_path)
