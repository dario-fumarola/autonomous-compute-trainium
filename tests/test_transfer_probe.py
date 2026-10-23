import math

import numpy as np
import torch

from scripts.run_transfer_probe import readout_fit


def test_readout_starts_at_original_predictor_and_learns_held_out_features():
    torch.set_num_threads(2)
    labels=torch.arange(64)%4
    features=torch.nn.functional.one_hot(labels,8).float()
    original=torch.zeros(64,4)
    pristine=features.clone()
    train=torch.arange(32);evaluation=torch.arange(32,64)
    result=readout_fit(features,original,labels,train,evaluation,.03,101)
    assert np.allclose(result[0]['losses'],math.log(4))
    assert result[512]['losses'].mean()<.15
    assert result[512]['correct'].all()
    torch.testing.assert_close(features,pristine)
    repeated=readout_fit(features,original,labels,train,evaluation,.03,101)
    np.testing.assert_array_equal(result[512]['losses'],repeated[512]['losses'])
