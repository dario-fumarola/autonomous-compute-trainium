import pytest
import torch
from compute_trainium.prefix_error_workspace import (
    read_weights, read_correction, observed_errors, rotated_error,
    sequential_bias_reference,
)


def fixture():
    torch.manual_seed(132)
    h = torch.randn(2, 9, 7, dtype=torch.double)
    logits = torch.randn(2, 9, 13, dtype=torch.double)
    y = torch.randint(13, (2, 9))
    return h, logits, y


@pytest.mark.parametrize('temperature', [0., 16.])
def test_literal_weighted_gradient_and_prefix_causality(temperature):
    h, logits, y = fixture()
    q = torch.tensor([0, 1, 4, 8])
    weights = read_weights(h, q, temperature=temperature)
    actual = read_correction(logits, y, weights)
    expected = sequential_bias_reference(logits, y, weights, q)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    assert actual[:, 0].count_nonzero() == 0
    for end in q.tolist():
        anchor = torch.tensor([end])
        w = read_weights(h, anchor, temperature=temperature)
        changed_y = y.clone(); changed_y[:, end:] = (changed_y[:, end:] + 1) % 13
        changed_logits = logits.clone(); changed_logits[:, end:] += torch.arange(13.)
        changed_h = h.clone(); changed_h[:, end+1:] += 100.
        for mode in ['error', 'label', 'prediction']:
            original = read_correction(logits, y, w, mode=mode)
            altered = read_correction(changed_logits, changed_y,
                       read_weights(changed_h, anchor, temperature=temperature), mode=mode)
            torch.testing.assert_close(original, altered, atol=0, rtol=0)
        torch.testing.assert_close(rotated_error(logits, y, w, anchor),
                                   rotated_error(changed_logits, changed_y, w, anchor))


def test_components_normalization_and_zero_identity():
    h, logits, y = fixture(); q = torch.tensor([0, 4, 8]); w = read_weights(h, q)
    error = read_correction(logits, y, w)
    label = read_correction(logits, y, w, mode='label')
    prediction = read_correction(logits, y, w, mode='prediction')
    torch.testing.assert_close(error, label+prediction, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(error.sum(-1), torch.zeros(2, 3, dtype=torch.double), atol=1e-14, rtol=0)
    assert torch.equal(logits[:, q] + 0 * error, logits[:, q])
    torch.testing.assert_close((logits[:, q] + error).softmax(-1).sum(-1), torch.ones(2, 3, dtype=torch.double))


def test_saturated_correct_class_retains_error_and_uniform_rotation():
    logits = torch.tensor([[[30., 0., -1.]]]); y = torch.tensor([[0]])
    r = observed_errors(logits, y)
    assert r[0, 0, 0] > 0
    torch.testing.assert_close(r.sum(-1), torch.zeros(1, 1, dtype=torch.double), atol=1e-19, rtol=0)
    h, logits, y = fixture(); q = torch.tensor([0, 4, 8]); w = read_weights(h, q, temperature=0)
    torch.testing.assert_close(rotated_error(logits, y, w, q), read_correction(logits, y, w), atol=1e-14, rtol=1e-12)


def test_large_vocabulary_fp32_inputs_against_double_literal_gradient():
    torch.manual_seed(133)
    logits = 3*torch.randn(1, 7, 100352)
    y = torch.randint(100352, (1, 7)); q = torch.tensor([3, 6])
    weights = read_weights(torch.randn(1, 7, 16), q)
    actual = read_correction(logits, y, weights)
    expected = sequential_bias_reference(logits.double(), y, weights.double(), q)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
