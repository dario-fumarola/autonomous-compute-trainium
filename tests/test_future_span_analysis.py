import pytest

from compute_trainium.future_span_analysis import point_contrasts,interaction_group_interval,pilot_decision
from compute_trainium.metrics import DocumentScore


def test_interaction_does_not_mistake_equal_objective_improvements_or_damage_for_success():
    values={('one-ntp',1):3.,('one-future',1):2.9,('recurrent-ntp',2):2.95,
        ('recurrent-future',2):2.85,('recurrent-future',1):2.87}
    equal=point_contrasts(values)
    assert equal['objective_recurrence_interaction']==pytest.approx(0.)
    assert not pilot_decision({5231:equal,7283:equal})['both_seeds_clear_all_point_gates']
    damaged=point_contrasts({**values,('one-future',1):3.2,('recurrent-future',2):2.96,('recurrent-future',1):3.})
    assert damaged['objective_recurrence_interaction']>.01
    assert not pilot_decision({5231:damaged,7283:damaged})['both_seeds_clear_all_point_gates']
    good=point_contrasts({**values,('recurrent-future',2):2.80})
    assert pilot_decision({5231:good,7283:good})['both_seeds_clear_all_point_gates']
    with pytest.raises(ValueError):pilot_decision({5231:good})


def test_four_cell_bootstrap_retains_original_denominator_and_pairing():
    means={'one-ntp':3.,'one-future':2.9,'recurrent-ntp':2.95,'recurrent-future':2.80}
    docs={cell:[DocumentScore(domain,f'{domain}{i}',f'{domain}{i}',value*10,10)
        for domain in ['web','math','code'] for i in range(8)] for cell,value in means.items()}
    result=interaction_group_interval(docs,draws=200,seed=11)
    assert result['gain_nats']==pytest.approx(.05)
    assert result['interval_95']==pytest.approx([.05,.05])
    docs['one-future'][0].tokens+=1
    with pytest.raises(ValueError):interaction_group_interval(docs,draws=200,seed=11)
