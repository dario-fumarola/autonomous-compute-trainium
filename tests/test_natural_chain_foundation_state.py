import pytest
from compute_trainium.natural_chain_foundation_state import foundation_initial


def test_each_history_selects_its_own_body_and_common_unseen_source_slice():
    p=dict(ntp_start_cursor=1210880,initial_states={role:dict(role=role,file=role+'-body.pt',parent_cursor=cursor) for role,cursor in [('one',1210880),('recurrent',983040)]})
    for role in ['one','recurrent']:
        assert foundation_initial(p,dict(initial=role,arm=role))['file']==role+'-body.pt'
    with pytest.raises(ValueError):foundation_initial(p,dict(initial='one',arm='recurrent'))
    p['initial_states']['one_source']=dict(role='one',file='one-source-body.pt',parent_cursor=983040)
    assert foundation_initial(p,dict(initial='one_source',arm='one'))['file']=='one-source-body.pt'
    p['ntp_start_cursor']=983040
    with pytest.raises(ValueError):foundation_initial(p,dict(initial='one',arm='one'))
    p['ntp_start_cursor']=1210880;p['initial_states']['one']['file']='../outside.pt'
    with pytest.raises(ValueError):foundation_initial(p,dict(initial='one',arm='one'))
