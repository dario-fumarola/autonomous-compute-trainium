import pytest
from compute_trainium.code_context_views import views,summarize


def test_short_view_preserves_targets_positions_and_noise_slice_alignment():
    # Integer token coordinates are software fixtures,not research data.
    tokens=list(range(1200));row={'target_start_token':1100,'target_end_token':1103,
        'full_start_token':78,'short_start_token':972}
    result=views(tokens,row);a,b=result['full'],result['short']
    assert len(a['inputs'])==1024 and len(b['inputs'])==130
    assert a['targets']==b['targets']==[1100,1101,1102]
    assert a['positions'][b['state_start']:]==b['positions']
    assert [a['inputs'][p] for p in a['prediction_positions']]==[1099,1100,1101]
    assert [b['inputs'][p] for p in b['prediction_positions']]==[1099,1100,1101]
    with pytest.raises(ValueError):views(tokens,{**row,'short_start_token':971})


def test_positive_interaction_alone_does_not_imply_full_context_gain():
    rows=[{'band':b,'real_targets':2,'nll_sum':{'full1':3.,'full2':4.,'short1':2.,'short2':6.}}
          for b in ['129-256','257-512','513-900']]
    score=summarize(rows)
    assert score['full_gain']==-.5 and score['interaction']==1.5
