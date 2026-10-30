import pytest
import torch
from torch.nn import functional as F
from compute_trainium.model import ModelConfig
from compute_trainium.interleaved_workspace import InterleavedWorkspace, matrix_work as full_work
from compute_trainium.latent_credit_diagnostic import jacobi_credit, matrix_work


def fixture():
    torch.manual_seed(274)
    c = ModelConfig(dim=8, layers=1, heads=2, kv_heads=1, ffn_dim=16, vocab_size=13, max_seq=6)
    return InterleavedWorkspace(c).double(), torch.tensor([[1, 3, 5, 2]])


@pytest.mark.parametrize('rounds', [2, 3])
def test_full_matches_original_outputs_and_complete_gradients(rounds):
    model, x = fixture()
    original = model.jacobi(x, rounds)
    original.square().mean().backward()
    gradient = {n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    got = jacobi_credit(model, x, rounds, 'all')
    torch.testing.assert_close(got, original, atol=1e-12, rtol=1e-12)
    got.square().mean().backward()
    for n,p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradient[n], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('retained', [0, 1])
def test_forward_unchanged_and_suffix_gradient_matches_frozen_prefix(retained):
    model, x = fixture()
    full = model.jacobi(x, 3)
    got, state = jacobi_credit(model, x, 3, retained, return_state=True)
    torch.testing.assert_close(got, full, atol=1e-12, rtol=1e-12)
    assert state.requires_grad == bool(retained)
    got.square().mean().backward()
    gradient = {n:p.grad.clone() for n,p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        source = model.embeddings(x)
        positions = torch.arange(x.shape[1])
        frozen = model.body_at(source, positions)
        for _ in range(3-retained):
            frozen = model.body_at(model.interleave(source, frozen), positions.repeat_interleave(2))[:, ::2]
    def with_frozen_prefix():
        source = model.embeddings(x)
        z = frozen
        for _ in range(retained):
            z = model.body_at(model.interleave(source, z), positions.repeat_interleave(2))[:, ::2]
        final = model.body_at(model.interleave(source, z), positions.repeat_interleave(2))[:, 1::2]
        return F.linear(final, model.prediction_weight).square().mean()
    with_frozen_prefix().backward()
    for n,p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradient[n], atol=1e-12, rtol=1e-12)
    p = model.blocks[0].up.weight
    expected = p.grad[0, 0].item()
    with torch.no_grad():
        old = p[0,0].item(); eps = 1e-5
        p[0,0] = old+eps; plus = with_frozen_prefix().item()
        p[0,0] = old-eps; minus = with_frozen_prefix().item()
        p[0,0] = old
    assert (plus-minus)/(2*eps) == pytest.approx(expected, abs=1e-7, rel=1e-4)


def test_work_keeps_every_forward_and_charges_retained_backward():
    model, _ = fixture()
    full = matrix_work(model.config, 4, 3, 'all')
    assert full['training_matmul_flops'] == full_work(model.config, 4, 3)['training_matmul_flops']
    one = matrix_work(model.config, 4, 3, 1)
    zero = matrix_work(model.config, 4, 3, 0)
    assert full['forward_matmul_flops'] == one['forward_matmul_flops'] == zero['forward_matmul_flops']
    assert full['training_matmul_flops'] > one['training_matmul_flops'] > zero['training_matmul_flops']


def test_bad_credit_budgets_rejected():
    model, x = fixture()
    for k in [-1, True, 3, 'none']:
        with pytest.raises(ValueError):
            jacobi_credit(model, x, 3, k)
