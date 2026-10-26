import pytest
import torch
from compute_trainium.lean_mixing_precision import LeanFP32MixingCandidate
from compute_trainium.parcae_candidate import ParcaeCandidateConfig


def test_lean_mixing_retains_exact_prediction_inventory_without_auxiliary_path():
    with torch.device('meta'):
        model=LeanFP32MixingCandidate(ParcaeCandidateConfig(vocab_size=32000),initialize=False)
    assert sum(p.numel() for p in model.parameters())==141963840
    assert not any(k.startswith('future_lm_head.') or k=='future_positions' for k in model.state_dict())
    assert model.precision_mode=='mix32'


def test_cheap_control_cannot_silently_become_ordinary():
    with pytest.raises(ValueError):LeanFP32MixingCandidate(ParcaeCandidateConfig(),recurrent=False,initialize=False)
