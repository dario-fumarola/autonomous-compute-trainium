import pytest
from compute_trainium.conclusion_generation import GreedyConclusion,correct_scalar

def test_generation_uses_visible_prefix_only_and_stops_on_eos_or_fixed_budget():
    run=GreedyConclusion((1,5,6),context=6,maximum_new_tokens=3)
    assert run.input_ids()==[1,5,6,2,2,2] and run.prediction_position==2
    run.advance(7);assert run.input_ids()==[1,5,6,7,2,2] and run.prediction_position==3
    run.advance(8);run.advance(9);assert run.done and not run.ended
    with pytest.raises(ValueError):run.advance(2)
    early=GreedyConclusion((1,5),context=6);early.advance(2);assert early.done and early.ended and early.generated==[]

def test_scalar_accuracy_requires_termination_and_checks_whole_generated_answer():
    assert correct_scalar(' -6/9\n','-2/3',True)
    assert not correct_scalar('45','45',False)
    assert not correct_scalar('45\nthen more','45',True)
    assert not correct_scalar('','45',True)
    assert not correct_scalar('40+5','45',True)
