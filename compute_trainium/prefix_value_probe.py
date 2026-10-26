"""Causal lag features and source-group cross-fitting for an offline probe."""
import hashlib

import numpy as np


def group_fold(group, folds=5):
    if not isinstance(group, str) or not group or folds < 2:
        raise ValueError('A real source group and at least two folds are required')
    key = b'past-surprisal-probe-v1\0'+group.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % folds


def lag_features(depth_one_losses, frames, context=1024, clip=20.):
    """Row t uses only earlier depth-one losses in the same causal window."""
    losses = np.asarray(depth_one_losses)
    if losses.ndim != 1 or not np.isfinite(losses).all() or (losses < 0).any() or context < 2:
        raise ValueError('Finite nonnegative losses and a causal context required')
    features = np.zeros((len(losses), 5), dtype=np.float64)
    cursor = 0
    for frame in frames:
        start, length = frame['offset'], frame['length']
        if start != cursor or length <= 0 or start+length > len(losses):
            raise ValueError('Exact contiguous document coverage required')
        cursor += length
        for left in range(start, start+length, context):
            right = min(left+context, start+length)
            values = np.clip(losses[left:right], 0., clip)
            prefix = np.concatenate(([0.], np.cumsum(values)))
            position = np.arange(len(values))
            features[left:right, 0] = np.concatenate(([0.], values[:-1]))
            for column, lag in [(1, 4), (2, 16)]:
                count = np.minimum(position, lag)
                features[left:right, column] = (prefix[position]-prefix[position-count])/np.maximum(count, 1)
            features[left:right, 3] = position/(context-1)
            features[left:right, 4] = position > 0
    if cursor != len(losses):
        raise ValueError('Every original target must remain covered')
    return features


def crossfit_probe(features, losses, frames, *, folds=5, ridge=1e-3,
                   weights=None):
    """Fit depth1/2 loss differences to depth3, separately in each training fold.

    Domain/group metadata control splitting and evaluation weights only; they
    are not predictor inputs. No current target loss is a predictor feature.
    """
    weights = weights or {'web': .6, 'math': .2, 'code': .2}
    x, losses = np.asarray(features), np.asarray(losses)
    if (x.ndim != 2 or losses.shape != (3, len(x)) or len(x) == 0
            or not np.isfinite(x).all() or not np.isfinite(losses).all()
            or (losses < 0).any() or ridge <= 0):
        raise ValueError('Finite paired features/losses and positive fixed ridge required')
    domain_order = list(weights)
    fold = np.full(len(x), -1, dtype=int)
    domains = np.full(len(x), -1, dtype=int)
    cursor = 0
    for frame in frames:
        if frame['offset'] != cursor or frame['length'] <= 0:
            raise ValueError('Original document coverage required')
        end = cursor+frame['length']
        if end > len(x) or frame['domain'] not in weights:
            raise ValueError('Original domain/target coverage required')
        fold[cursor:end] = group_fold(frame['group'], folds)
        domains[cursor:end] = domain_order.index(frame['domain'])
        cursor = end
    if cursor != len(x) or set(fold) != set(range(folds)):
        raise ValueError('All fixed source folds must be nonempty')
    scores = np.full((len(x), 3), np.nan)
    constant = np.full(len(x), -1, dtype=int)
    records = []
    for heldout in range(folds):
        train, test = fold != heldout, fold == heldout
        sample_weight = np.zeros(int(train.sum()))
        training_domains = domains[train]
        for index, domain in enumerate(domain_order):
            count = int((training_domains == index).sum())
            if not count or not (domains[test] == index).any():
                raise ValueError('Every training/validation fold needs every domain')
            sample_weight[training_domains == index] = weights[domain]/count
        sample_weight /= sample_weight.sum()
        mean = sample_weight @ x[train]
        scale = np.sqrt(sample_weight @ ((x[train]-mean)**2))
        scale = np.where(scale > 1e-12, scale, 1.)
        a = np.column_stack((np.ones(train.sum()), (x[train]-mean)/scale))
        b = np.column_stack((np.ones(test.sum()), (x[test]-mean)/scale))
        y = (losses[:2, train]-losses[2, train]).T
        penalty = np.eye(a.shape[1])*ridge; penalty[0, 0] = 0.
        coef = np.linalg.solve(a.T @ (sample_weight[:, None]*a)+penalty,
                               a.T @ (sample_weight[:, None]*y))
        scores[test, :2] = b @ coef; scores[test, 2] = 0.
        chosen_constant = int(np.argmin(losses[:, train] @ sample_weight))
        constant[test] = chosen_constant
        records.append(dict(heldout_fold=heldout, training_targets=int(train.sum()),
            validation_targets=int(test.sum()), fitted_coefficients=int(coef.size),
            training_only_constant_depth=chosen_constant+1,
            coefficient_l2=float(np.linalg.norm(coef))))
    if not np.isfinite(scores).all() or (constant < 0).any():
        raise AssertionError('Cross-fitting did not cover every target')
    return dict(predicted_differences=scores, choices=np.argmin(scores, axis=1),
                constant_choices=constant, folds=fold, fit_records=records)
