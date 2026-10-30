import pytest
from compute_trainium.calculation_exposure import intact_exposure


def row(start=0,p=(0,3),t=(4,6),key='x'):
    return dict(start=start,premise_tokens=p,target_tokens=t,operations=1,expression_key=key)


def test_full_window_and_exposure_end_are_half_open():
    assert intact_exposure([row()],7,8)['intact_answer_tokens']==2
    assert intact_exposure([row()],6,8)['intact_answer_tokens']==0
    assert intact_exposure([row(t=(6,8))],9,8)['intact_answer_tokens']==2
    assert intact_exposure([row(t=(7,9))],10,8)['intact_answer_tokens']==0


def test_premise_must_remain_visible_and_document_boundaries_are_accounted():
    assert intact_exposure([row(t=(8,10))],20,8)['intact_answer_tokens']==0
    assert intact_exposure([row(start=8)],20,8)['intact_answer_tokens']==2
    x=intact_exposure([row(),row(start=8)],20,8)
    assert x==dict(intact_calculation_counts={'1':2},intact_answer_tokens=4,distinct_expressions=1)
    with pytest.raises(ValueError):intact_exposure([row(p=(4,5),t=(1,2))],20)
