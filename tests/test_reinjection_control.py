"""Numerical fixtures only; never synthetic research data."""
import pytest
import torch

from compute_trainium.model import ModelConfig
from compute_trainium.reinjection_control import (ReinjectionControlConfig, reinjection_unroll,
                                                 IndependentReinjectionControl)


def test_injection_control_has_one_call_per_block_and_two_gradient_paths():
    c = ReinjectionControlConfig(dim=8, layers=4, heads=2, kv_heads=1,
                                ffn_dim=16, vocab_size=17, prelude=1, core=2, coda=1)
    scales = [torch.tensor(float(i+1), requires_grad=True) for i in range(4)]
    x = torch.tensor(2., requires_grad=True)
    calls = []
    def apply(scale, value, index):
        calls.append(index)
        return scale*value
    y = reinjection_unroll(x, scales, c, apply)
    y.backward()
    assert calls == [0, 1, 2, 3]
    assert y.item() == 56
    assert x.grad.item() == 28
    # The prelude receives the main path and the separate bypass gradient.
    assert scales[0].grad.item() == 56
    assert all(s.grad is not None and torch.isfinite(s.grad) for s in scales)
    ordinary = ModelConfig(**{k: getattr(c, k) for k in ModelConfig.__dataclass_fields__})
    assert c.parameter_count() == ordinary.parameter_count()
    assert c.flops(7, 7)['training_matmul_flops'] == ordinary.flops(7, 7)['training_matmul_flops']


def test_control_marker_rejects_an_accidental_second_core_application():
    with pytest.raises(ValueError, match='once'):
        ReinjectionControlConfig(repetitions=2)
    with pytest.raises(ValueError, match='once'):
        ReinjectionControlConfig(reinjection_only=False)


def test_independent_control_preserves_causality_and_trains_every_block():
    torch.manual_seed(619)
    c = ModelConfig(dim=8, layers=4, heads=2, kv_heads=1, ffn_dim=16, vocab_size=17)
    model = IndependentReinjectionControl(c, prelude=1, core=2, coda=1)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    changed = tokens.clone()
    changed[:, 3:] = torch.tensor([[6, 7]])
    original = model(tokens)
    torch.testing.assert_close(original[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    original.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == c.parameter_count()


def test_cpu_fixture_maps_qkv_gradients_by_kv_group():
    from scripts.verify_reinjection_factory import grouped_qkv_gradient
    q = torch.tensor([[1.], [2.], [3.], [4.]])
    k, v = torch.tensor([[5.], [6.]]), torch.tensor([[7.], [8.]])
    expected = torch.tensor([[1.], [2.], [5.], [7.], [3.], [4.], [6.], [8.]])
    torch.testing.assert_close(grouped_qkv_gradient(q, k, v, 2, 1), expected)
