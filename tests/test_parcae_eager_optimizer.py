import pytest
import torch

from scripts.capture_parcae_optimizer_eager import eager_optimizer


def test_eager_diagnostic_changes_execution_only_and_preserves_algorithm():
    model=torch.nn.Linear(8,12,bias=False)
    optimizer=eager_optimizer(model,gate_policy='release_muon',fused_adamw=True,muon_backend='neuron')
    assert optimizer.muon.tensor_step.__name__=='muon_tensor_step'
    assert optimizer.recipe['specified_muon_polar_dtype']=='bf16'
    with pytest.raises(ValueError):
        eager_optimizer(model,gate_policy='release_muon',fused_adamw=True,muon_backend=None)
