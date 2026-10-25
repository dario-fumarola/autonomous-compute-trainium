import numpy as np
import pytest
import torch
from torch.nn import functional as F

from compute_trainium.future_span import FutureSpanConfig, future_span_targets
from compute_trainium.future_span_frequency import (
    presence_counts, probabilities, constant_presence_loss, count_source_spans,
)


def test_frequency_reference_agrees_with_dense_binary_presence_loss():
    bags = np.array([[1, 1, 2], [1, 3, 3], [4, 4, 4]])
    counts, anchors = presence_counts(bags, np.array([True, True, False]), 5)
    assert anchors == 2 and counts.tolist() == [0, 2, 1, 1, 0]
    probs = probabilities(counts, anchors)
    heldout = np.array([[1, 1, 4], [2, 3, 4]])
    target_counts, size = presence_counts(heldout, np.ones(2, dtype=bool), 5)
    labels = torch.zeros((2, 5), dtype=torch.float64).scatter_(1, torch.tensor(heldout), 1)
    logits = torch.tensor(np.log(probs)-np.log1p(-probs)).expand_as(labels)
    expected = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum').item()/2/3
    assert constant_presence_loss(target_counts, size, probs, 4) == pytest.approx(expected, abs=1e-14)
    for wrong in [np.zeros(5), np.full(5, np.nan), np.ones(5)]:
        with pytest.raises(ValueError): constant_presence_loss(target_counts, size, wrong, 4)


def test_source_counting_preserves_window_document_and_immediate_target_rules():
    # Numerical boundary fixtures only;never used as research source text.
    config = FutureSpanConfig(context=16, horizon=4, stride=2)
    tokens = np.arange(49, dtype=np.int32) % 13
    domains = np.ones(49, dtype=np.uint8)
    domains[[0, 9, 17, 31, 43]] = 0
    actual, size = count_source_spans(tokens, domains, start=0, stop=3, context=16,
        horizon=4, stride=2, vocab_size=13)
    expected = np.zeros(13, dtype=np.int64);total = 0
    for window in range(3):
        base = window*16
        labels = future_span_targets(torch.tensor(tokens[None, base:base+17], dtype=torch.long),
            torch.tensor(domains[None, base:base+17] > 0), config, 13)
        current, count = presence_counts(labels.token_ids.numpy().reshape(-1, 3),
            labels.valid.numpy().reshape(-1), 13)
        expected += current;total += count
    np.testing.assert_array_equal(actual, expected)
    assert total == size
    with pytest.raises(ValueError):
        count_source_spans(tokens, domains, start=0, stop=4, context=16, horizon=4, stride=2, vocab_size=13)
