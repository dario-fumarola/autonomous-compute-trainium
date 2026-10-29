"""Numerical fixtures only, never generated research training examples."""
import copy

import pytest
import torch

from compute_trainium.depth_causal_workspace import DepthCausalWorkspace, depth_mask, matrix_work
from compute_trainium.interleaved_workspace import InterleavedWorkspace
from compute_trainium.model import ModelConfig


def make(dtype=torch.float64, explicit=False):
    torch.manual_seed(631)
    c = ModelConfig(dim=16, layers=2, heads=4, kv_heads=2, ffn_dim=24, vocab_size=31, max_seq=8)
    return DepthCausalWorkspace(c, explicit_attention=explicit).to(dtype)


TOKENS = torch.tensor([[2, 5, 8, 11, 14], [3, 6, 9, 12, 15]])


def loss(logits):
    labels = (TOKENS+1) % logits.shape[-1]
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())


def compare_grads(a, b, atol, rtol):
    assert a.state_dict().keys() == b.state_dict().keys()
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=atol, rtol=rtol, msg=name)


def test_two_dimensional_mask_has_no_future_or_deeper_memory():
    actual = depth_mask(4, 3)
    expected = torch.tensor([[s <= t and j <= k for j in range(3) for s in range(4)]
                             for k in range(3) for t in range(4)])
    assert torch.equal(actual, expected)
    assert actual[5, 1]  # stage1 t1 may use stage0 t1
    assert not actual[5, 2]  # later source at lower stage is still forbidden
    assert not actual[5, 8]  # earlier source at higher stage is forbidden


@pytest.mark.parametrize('budget', [0, 1, 2])
@pytest.mark.parametrize('dtype,explicit', [(torch.float64, False), (torch.float64, True), (torch.float32, True)])
def test_cached_full_outputs_states_and_all_gradients(budget, dtype, explicit):
    model = make(dtype, explicit)
    reference = copy.deepcopy(model)
    actual, states = model(TOKENS, budget, return_stages=True)
    expected, ref_states = reference.full_reference(TOKENS, budget, return_stages=True)
    atol, rtol = (1e-11, 1e-9) if dtype == torch.float64 else (4e-6, 2e-4)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for a, b in zip(states, ref_states):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    loss(actual).backward()
    loss(expected).backward()
    compare_grads(model, reference, atol, rtol)


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_token_major_inference_matches_parallel_values_and_gradients(budget):
    model, reference = make(), make()
    actual, states = model(TOKENS, budget, return_stages=True)
    expected, ref_states = reference.sequential_reference(TOKENS, budget, return_stages=True)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)
    for a, b in zip(states, ref_states):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
    loss(actual).backward()
    loss(expected).backward()
    compare_grads(model, reference, 1e-11, 1e-9)


def test_zero_matches_original_ordinary_through_adamw_step():
    model = make()
    torch.manual_seed(631)
    ordinary = InterleavedWorkspace(model.config).double()
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p, ordinary.state_dict()[name], rtol=0, atol=0)
    actual, expected = model(TOKENS, 0), ordinary(TOKENS, 0, 'ordinary')
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    loss(actual).backward()
    loss(expected).backward()
    compare_grads(model, ordinary, 1e-12, 1e-10)
    a = torch.optim.AdamW(model.parameters(), lr=.001)
    b = torch.optim.AdamW(ordinary.parameters(), lr=.001)
    a.step()
    b.step()
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p, ordinary.state_dict()[name], atol=1e-12, rtol=1e-10)
    for x, y in zip(a.state.values(), b.state.values()):
        for key in x:
            torch.testing.assert_close(x[key], y[key], atol=1e-12, rtol=1e-10)


def test_deeper_budget_cannot_change_previous_stages():
    model = make()
    full = model(TOKENS, 2, return_stages=True)[1]
    for budget in (0, 1):
        small = model(TOKENS, budget, return_stages=True)[1]
        for actual, expected in zip(small, full):
            assert torch.equal(actual, expected)


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_future_source_change_cannot_change_prefix_or_latents(budget):
    model = make()
    altered = TOKENS.clone()
    altered[:, 3:] = 29
    a, states_a = model(TOKENS, budget, return_stages=True)
    b, states_b = model(altered, budget, return_stages=True)
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=0, rtol=0)
    for x, y in zip(states_a, states_b):
        torch.testing.assert_close(x[:, :3], y[:, :3], atol=0, rtol=0)


def test_detaching_cache_is_forward_equivalent_but_wrong_gradient():
    model, broken = make(), make()
    a, b = model(TOKENS, 2), broken(TOKENS, 2, detach_cache=True)
    assert torch.equal(a, b)
    loss(a).backward()
    loss(b).backward()
    delta = sum((p.grad-q.grad).square().sum() for p, q in zip(model.parameters(), broken.parameters())).sqrt()
    assert delta.item() > 1e-4
    assert (model.blocks[0].v.weight.grad-broken.blocks[0].v.weight.grad).abs().max() > 1e-5


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_ledger_matches_each_executed_projection_and_attention_rectangle(budget):
    c = make().config
    n, batch = 5, 2
    projections = attention = 0
    for stage in range(budget+1):
        for _ in range(c.layers):
            kv = c.kv_heads*c.dim//c.heads
            # q,k,v,o and gate,up,down; each projection occurs once per stage.
            projections += 2*batch*n*(c.dim*c.dim+2*c.dim*kv+c.dim*c.dim+3*c.dim*c.ffn_dim)
            attention += 4*batch*n*((stage+1)*n)*c.dim
    head = 2*batch*n*c.dim*c.vocab_size
    actual = matrix_work(c, n, budget, batch)
    assert actual['projection_forward_flops'] == projections
    assert actual['attention_forward_flops'] == attention
    assert actual['training_matmul_flops'] == 3*(projections+attention+head)
    if budget == 0:
        assert actual['forward_matmul_flops'] == c.flops(n, n, batch)['forward_matmul_flops']
    else:
        assert actual['forward_matmul_flops'] < matrix_work(c, n, budget, batch, full_recomputation=True)['forward_matmul_flops']


@pytest.mark.parametrize('bad', [-1, 3, True, 1.0])
def test_invalid_budget_fails_closed(bad):
    with pytest.raises(ValueError):
        make()(TOKENS, bad)


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_qualification_factory_uses_independent_cpu_and_native_paths(budget):
    from compute_trainium.depth_causal_factory import make_model, matrix_work as factory_work
    c = make().config
    plan = dict(model_config=c.__dict__, workspace_mechanism='depth_causal_generated_state')
    torch.manual_seed(833)
    native = make_model(plan, explicit_attention=True).double()
    torch.manual_seed(833)
    cpu = make_model(plan).double()
    a, b = native(TOKENS, budget), cpu(TOKENS, budget)
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
    loss(a).backward()
    loss(b).backward()
    compare_grads(native, cpu, 1e-11, 1e-9)
    assert factory_work(c, 5, budget, batch=2) == matrix_work(c, 5, budget, 2)


def test_qualification_factory_rejects_wrong_mechanism():
    from compute_trainium.depth_causal_factory import make_model
    with pytest.raises(ValueError):
        make_model(dict(model_config=make().config.__dict__, workspace_mechanism='budgeted_persistent_generated_state'))
