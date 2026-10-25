"""Fixed TRAIN-only frequency reference for the existing presence objective."""
import numpy as np


def presence_counts(token_bags, valid, vocab_size):
    bags, valid = np.asarray(token_bags), np.asarray(valid)
    if (bags.ndim != 2 or valid.shape != bags.shape[:1] or valid.dtype != np.bool_
            or not np.issubdtype(bags.dtype, np.integer) or vocab_size < 2
            or bags.shape[1] < 1 or np.any(bags < 0) or np.any(bags >= vocab_size)):
        raise ValueError('Aligned valid bags of in-vocabulary integer targets required')
    selected = np.sort(bags[valid], axis=1)
    unique = np.ones(selected.shape, dtype=bool)
    unique[:, 1:] = selected[:, 1:] != selected[:, :-1]
    return np.bincount(selected[unique], minlength=vocab_size), int(valid.sum())


def probabilities(counts, anchors):
    values = np.asarray(counts)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or anchors < 1 or np.any(values < 0) or np.any(values > anchors)):
        raise ValueError('Integer presence counts and positive anchor total required')
    # One fixed Jeffreys Beta prior per binary label;no tuning or model labels.
    return (values.astype(np.float64)+.5)/(anchors+1)


def constant_presence_loss(counts, anchors, probs, horizon):
    counts, probs = np.asarray(counts), np.asarray(probs)
    probabilities(counts, anchors)  # Validate counts independently of prediction.
    if (counts.shape != probs.shape or horizon < 2 or not np.isfinite(probs).all()
            or np.any(probs <= 0) or np.any(probs >= 1)):
        raise ValueError('Finite,strictly interior constant label probabilities required')
    return float((-counts*np.log(probs)-(anchors-counts)*np.log1p(-probs)).sum()
                 /anchors/(horizon-1))


def count_source_spans(tokens, domains, *, start, stop, context, horizon, stride, vocab_size,
                       guard=lambda: None):
    if not 0 <= start < stop or (stop*context+1 > len(tokens)) or len(tokens) != len(domains):
        raise ValueError('A finite original nonwrapping source interval is required')
    if not 2 <= horizon <= context or stride < 1:
        raise ValueError('Original finite horizon and stride required')
    positions = np.arange(0, context-horizon+1, stride)
    target_positions = positions[:, None]+np.arange(2, horizon+1)[None, :]
    span_positions = positions[:, None]+np.arange(horizon+1)[None, :]
    counts, anchors = np.zeros(vocab_size, dtype=np.int64), 0
    for first in range(start, stop, 32):
        guard()
        indices = np.arange(first, min(first+32, stop))[:, None]*context+np.arange(context+1)[None, :]
        source, real = np.asarray(tokens)[indices], np.asarray(domains)[indices] > 0
        labels = source[:, target_positions].reshape(-1, horizon-1)
        valid = real[:, span_positions].all(-1).reshape(-1)
        current, size = presence_counts(labels, valid, vocab_size)
        counts += current
        anchors += size
    return counts, anchors
