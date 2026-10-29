import pytest
from scripts.analyze_probability_fixed_one import repeat_error
from compute_trainium.model import ModelConfig
from compute_trainium.probability_workspace import matrix_work

def test_fixed_one_matches_balanced_sampled_work():
    c=ModelConfig(dim=512,layers=16,heads=8,kv_heads=4,ffn_dim=1376,vocab_size=32000,max_seq=512)
    work=lambda k:64*matrix_work(c,512,k,batch=4)['training_matmul_flops']
    assert 1530*work(1)==510*sum(work(k) for k in (0,1,2))==190988262121144320

def test_reused_scores_require_same_examples_and_targets():
    old=dict(label='ordinary-data-final-k0',panel='broad',id='fixture',group='group',targets=2,kind='broad',domain='web',loss_sum=4.,first_loss=1.5)
    new=dict(old,label='ordinary_data-k0')
    assert repeat_error([new],[old])==0
    assert repeat_error([dict(new,loss_sum=4.0002)],[old])==pytest.approx(.0001)
    with pytest.raises(AssertionError):repeat_error([dict(new,targets=3)],[old])
    with pytest.raises(AssertionError):repeat_error([dict(new,id='different')],[old])
