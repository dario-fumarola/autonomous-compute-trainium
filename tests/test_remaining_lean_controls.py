import copy
import pytest
from scripts.run_remaining_lean_controls import require_complete_control


def fixture_report():
    score=dict(status='COMPLETE_INDEPENDENT_LOW_DEPTH_DEVELOPMENT_SCORE',checkpoint_sha256='a'*64)
    gate={'independent_cpu':dict(status='PASS_INDEPENDENT_MATURE_LOW_DEPTH_GRADIENT',checkpoint_sha256='a'*64)}
    return dict(status='COMPLETE_QUALIFIED_LEAN_BASELINE_MAIN_CONTROL',model_variant='ordinary8',
        scores={s:copy.deepcopy(score) for s in ['5231','17293']},
        final_gradient_gates={s:copy.deepcopy(gate) for s in ['5231','17293']},
        recurrence_advantage={s:{'gain_nats':-.05} for s in ['5231','17293']})


def test_queue_keeps_negative_controls_but_never_advances_on_failed_audits():
    report=fixture_report();assert require_complete_control(report,'ordinary8')
    report['final_gradient_gates']['17293']['independent_cpu']['status']='FAIL'
    with pytest.raises(ValueError):require_complete_control(report,'ordinary8')


def test_queue_requires_both_frozen_seeds_and_exact_scored_checkpoint():
    report=fixture_report();report['scores'].pop('17293')
    with pytest.raises(ValueError):require_complete_control(report,'ordinary8')
    report=fixture_report();report['scores']['5231']['checkpoint_sha256']='b'*64
    with pytest.raises(ValueError):require_complete_control(report,'ordinary8')
