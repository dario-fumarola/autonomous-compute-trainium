import numpy as np
import pytest
from compute_trainium.reference_loss_strata import reference_deciles


def test_independent_bins_preserve_every_position_and_the_pooled_gain_with_ties():
    index=[dict(id='a',group='x',domain='web',offset=0,length=5),dict(id='b',group='y',domain='code',offset=5,length=4)]
    ref=np.array([0,0,1,2,3,0,1,1,2.]);base=np.arange(9,dtype=float);candidate=base-.02
    result=reference_deciles(ref,base,candidate,index,weights={'web':.6,'code':.4},bins=2)
    assert sum(r['targets'] for r in result['rows'])==9
    assert result['pooled_gain_nats']==pytest.approx(.02)
    assert result['reconstructed_pooled_gain_nats']==pytest.approx(.02)
    bad=[dict(index[0]),dict(index[1],offset=4)]
    with pytest.raises(ValueError,match='contiguous'):reference_deciles(ref,base,candidate,bad,weights={'web':.6,'code':.4},bins=2)
    with pytest.raises(ValueError,match='paired'):reference_deciles(ref[:-1],base,candidate,index,weights={'web':.6,'code':.4},bins=2)
