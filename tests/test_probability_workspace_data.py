import json
import numpy as np
import pytest
import torch

from compute_trainium.probability_workspace_data import ProbabilitySources


def test_real_targets_are_aligned_once_across_overlapping_input_windows(tmp_path):
    # Numerical fixture only, not research training text.
    np.save(tmp_path/'train-tokens.npy',np.array([1,5,6,7,2,1,8,9,2],dtype=np.uint16))
    np.save(tmp_path/'train-domains.npy',np.array([0,1,1,1,0,0,2,2,0],dtype=np.uint8))
    (tmp_path/'report.json').write_text(json.dumps(dict(status='PASS_PROBABILITY_WORKSPACE_NATURAL_SOURCES',context=4,windows=2)))
    data=ProbabilitySources(tmp_path);x,y,m=data.batch([0,1])
    assert torch.equal(x,torch.tensor([[1,5,6,7],[2,1,8,9]]))
    assert torch.equal(y,torch.tensor([[5,6,7,2],[1,8,9,2]]))
    assert y[m].tolist()==[5,6,7,8,9]
    assert data.target_count(0,2)==int(m.sum())==5
    assert data.target_count(0,1)+data.target_count(1,1)==5
    for invalid in ([],[-1],[2]):
        with pytest.raises(ValueError):data.batch(invalid)


def test_unaccepted_data_rejected(tmp_path):
    (tmp_path/'report.json').write_text(json.dumps(dict(status='STOPPED')))
    with pytest.raises(ValueError):ProbabilitySources(tmp_path)
