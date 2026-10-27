from collections import OrderedDict
import pytest
import torch
from compute_trainium.canonical_prediction_checkpoint import canonical_prediction_checkpoint
from compute_trainium.training import atomic_checkpoint
from scripts.qualify_native import digest


def test_ordered_state_serializes_to_same_pre_save_digest_and_preserves_tied_values(tmp_path):
    weight=torch.arange(12,dtype=torch.float32).reshape(3,4)
    model=OrderedDict([('table',weight),('head',weight),('buffer',torch.tensor([1+2j]))])
    reference=digest(dict(model));assert digest(model)!=reference
    value=dict(purpose='scale_retained_prediction_weights',model=model,model_digest=reference)
    file=tmp_path/'state.pt';atomic_checkpoint(file,canonical_prediction_checkpoint(value))
    read=torch.load(file,weights_only=False)
    assert digest(read['model'])==reference and type(read['model']) is dict
    assert read['model']['table'].data_ptr()==read['model']['head'].data_ptr()
    read['model']['table'][0,0]+=1
    with pytest.raises(AssertionError):canonical_prediction_checkpoint(read)
