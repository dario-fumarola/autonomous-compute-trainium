import numpy as np
import pytest

from compute_trainium.future_span import FutureSpanConfig
from compute_trainium.parcae_candidate import ParcaeCandidateConfig
from compute_trainium.parcae_mixing_control import mixing_only_ledger
from compute_trainium.mixing_control_science import matrix_work


def test_complete_control_work_equals_sum_of_executed_full_gradient_paths():
    c=ParcaeCandidateConfig(dim=32,heads=2,ffn_dim=64,vocab_size=41,max_seq=16)
    span=FutureSpanConfig(context=16,horizon=4,stride=2)
    counts=np.array([[[1,3],[2,5]],[[4,1],[3,2]]],dtype=np.int16)
    depths=np.stack([counts,counts],axis=-1)
    observed=matrix_work(c,span,depths,world=2,optimizer_per_replica=17)
    auxiliary=6*len(span.positions)*c.dim*c.vocab_size
    expected=[sum(mixing_only_ledger(c,16,total_steps=int(n),backward_cap=int(n))['training_matmul_flops']
                  +auxiliary for n in row.flatten())+34 for row in counts]
    assert observed.tolist()==expected
    bad=depths.copy();bad[0,0,0,1]=2
    with pytest.raises(ValueError):matrix_work(c,span,bad,world=2,optimizer_per_replica=17)
    with pytest.raises(ValueError):matrix_work(c,span,depths.astype(float),world=2,optimizer_per_replica=17)
    with pytest.raises(ValueError):matrix_work(c,span,depths,world=3,optimizer_per_replica=17)
