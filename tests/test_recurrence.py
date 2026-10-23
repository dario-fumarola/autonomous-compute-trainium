import copy

import pytest
import torch

from compute_trainium.recurrence import RecurrenceConfig, fixed_unroll


def small(repetitions=2, untied=False):
    return RecurrenceConfig(dim=8, layers=4, heads=2, kv_heads=1, ffn_dim=16,
                            vocab_size=32, max_seq=32, prelude=1, core=2, coda=1,
                            repetitions=repetitions, untied=untied)


def apply(layer, x, index):
    return torch.tanh(layer(x))


def test_unroll_matches_independent_expression_and_full_bptt():
    torch.manual_seed(7)
    layers = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False) for _ in range(4)])
    x = torch.randn(2, 5, 8, requires_grad=True)
    actual = fixed_unroll(x, layers, small(), apply)
    e = apply(layers[0], x, 0)
    s = apply(layers[2], apply(layers[1], e, 1), 2)
    s = apply(layers[2], apply(layers[1], s+e, 1), 2)
    expected = apply(layers[3], s, 3)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (x, *layers.parameters()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (x, *layers.parameters()))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)
        assert a.norm() > 0


def test_shared_gradients_equal_sum_of_untied_copies():
    torch.manual_seed(8)
    layers = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False) for _ in range(4)])
    untied = copy.deepcopy(layers)
    extra = torch.nn.ModuleList([copy.deepcopy(layers[1]), copy.deepcopy(layers[2])])
    x = torch.randn(1, 4, 8)
    a = fixed_unroll(x, layers, small(), apply)
    b = fixed_unroll(x, untied, small(untied=True), apply, extra)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for i in range(4):
        expected = untied[i].weight.grad
        if i in (1, 2):
            expected = expected+extra[i-1].weight.grad
        torch.testing.assert_close(layers[i].weight.grad, expected)


def test_single_pass_is_plain_layer_composition():
    layers = [torch.nn.Linear(8, 8) for _ in range(4)]
    x = torch.randn(2, 3, 8)
    y = x
    for layer in layers:
        y = apply(layer, y, 0)
    torch.testing.assert_close(fixed_unroll(x, layers, small(1), apply), y)


def test_resource_counts_include_depth_and_head():
    a, b, u = small(1), small(2), small(2, True)
    assert a.parameter_count() == b.parameter_count() < u.parameter_count()
    af, bf, uf = [c.flops(32, 32) for c in (a, b, u)]
    assert af['forward_matmul_flops'] < bf['forward_matmul_flops'] == uf['forward_matmul_flops']
    assert af['lm_head_forward_flops'] == bf['lm_head_forward_flops']
    with pytest.raises(ValueError):
        RecurrenceConfig(repetitions=0)
