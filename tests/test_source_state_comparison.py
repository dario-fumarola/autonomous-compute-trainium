import pytest
import torch

from scripts.qualify_source_continuation import compare_trees


def test_source_continuation_requires_matching_metadata_and_finite_close_tensors():
    value={'state':torch.ones(4),'step':[8,{'lr':.001}]}
    assert compare_trees(value,value)['bit_identical']
    close={'state':torch.ones(4)+1e-6,'step':[8,{'lr':.001}]}
    assert not compare_trees(value,close)['bit_identical']
    for bad in [{'state':torch.ones(4)*1.01,'step':value['step']},
                {'state':torch.ones(4),'step':[9,{'lr':.001}]},
                {'state':torch.full((4,),float('nan')),'step':value['step']}]:
        with pytest.raises(AssertionError):compare_trees(value,bad)
