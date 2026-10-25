from pathlib import Path
import pytest

from compute_trainium.mixing_scoring_recovery import score_arguments


def test_scoring_arguments_join_checkpoint_as_path_before_serialization():
    args=score_arguments(Path('/private/root'),Path('/private/root/plan.json'),
        'source-sha','case','native-fit','checkpoint-sha',Path('/private/root/score'))
    assert args[args.index('--checkpoint')+1]=='/private/root/runs/native-fit/checkpoint-u576.pt'
    assert args[args.index('--plan')+1]=='/private/root/plan.json'
    assert args[args.index('--depth')+1]=='2'
    assert all(isinstance(s,str) for s in args)


def test_scoring_recovery_rejects_a_quality_failure_or_unfinished_parent():
    from compute_trainium.mixing_scoring_recovery import completed_fits
    with pytest.raises(ValueError):
        completed_fits({'cases':{}},{'status':'COMPLETE_BOUNDED_CHEAP_MIXING_TUNE','error':'','scores':[]})
    with pytest.raises(ValueError):
        completed_fits({'cases':{}},{'status':'STOPPED_CHEAP_MIXING_TUNE',
            'error':"unsupported operand type(s) for +: 'PosixPath' and 'str'",'scores':[],
            'training_records':[],'gradient_gates':[]})
