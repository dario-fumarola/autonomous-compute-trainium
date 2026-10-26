"""Frozen prefix features and TRAIN-fitted value readouts; no LM updates."""
import numpy as np
import torch


def prefix_features(hidden, logits, *, context=1024):
    """A row sees its current first-pass state/distribution, never its target.

    Input tensors come from a causal model at the same positions. This helper
    accepts no labels, second-pass tensors, source IDs or document lengths.
    """
    if (hidden.ndim != 2 or logits.ndim != 2 or len(hidden) != len(logits)
            or not 0 < len(hidden) <= context or logits.shape[1] < 2):
        raise ValueError('Aligned first-pass states and vocabulary logits required')
    scalars = []
    for start in range(0, len(logits), 128):
        values = logits[start:start+128].float()
        logp = values.log_softmax(-1)
        p = logp.exp()
        entropy = -(p*logp).sum(-1)
        top = p.topk(2, dim=-1).values
        scalars.append(torch.stack((entropy, top[:, 0]-top[:, 1]), -1))
    position = torch.arange(len(hidden), device=hidden.device).float() / (context-1)
    result = torch.cat((hidden.float(), torch.cat(scalars), position[:, None]), -1)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite prefix features')
    return result


def domain_weights(domains):
    labels = np.asarray(domains)
    weights = np.zeros(len(labels), dtype=np.float64)
    for key, mass in [(1, .6), (2, .2), (3, .2)]:
        count = int((labels == key).sum())
        if not count:
            raise ValueError('Every source domain must be represented')
        weights[labels == key] = mass/count
    if not np.all(weights > 0):
        raise ValueError('Only real source targets may enter the probe')
    return weights


def fit_readouts(features, gain, domains, *, ridge=.01, epochs=8,
                 hidden_width=64, batch_size=2048, seed=31415):
    """One fixed ridge fit and one fixed small MLP, using TRAIN labels only."""
    x = torch.as_tensor(np.array(features, dtype=np.float32))
    y = torch.as_tensor(np.array(gain, dtype=np.float32))
    if (x.ndim != 2 or y.shape != (len(x),) or not len(x)
            or not torch.isfinite(x).all() or not torch.isfinite(y).all()
            or ridge <= 0 or epochs < 1 or batch_size < 1):
        raise ValueError('Finite TRAIN features/labels and fixed fit settings required')
    w = torch.from_numpy(domain_weights(domains))
    mean = (w[:, None]*x).sum(0)
    scale = ((w[:, None]*(x-mean).square()).sum(0)).sqrt().clamp_min(1e-6)
    x = ((x-mean)/scale).float()
    # Accumulate in double precision; never normalize using calibration/DEV.
    width = x.shape[1]+1
    gram = torch.eye(width, dtype=torch.float64)*ridge
    gram[0, 0] = 0.
    rhs = torch.zeros(width, dtype=torch.float64)
    for left in range(0, len(x), 8192):
        right = min(left+8192, len(x))
        a = torch.cat((torch.ones(right-left, 1), x[left:right]), 1).double()
        weighted = a*w[left:right, None]
        gram.add_(a.T @ weighted)
        rhs.add_(weighted.T @ y[left:right].double())
    coefficient = torch.linalg.solve(gram, rhs)
    # Local RNG state avoids modifying the frozen language model's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mlp = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden_width),
            torch.nn.GELU(), torch.nn.Linear(hidden_width, 1))
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=.001, weight_decay=.001)
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for left in range(0, len(x), batch_size):
            indices = order[left:left+batch_size]
            error = mlp(x[indices]).squeeze(-1)-y[indices]
            loss = (error.square()*w[indices].float()*len(x)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite TRAIN probe objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.)
            optimizer.step(); updates += 1
    return dict(mean=mean, scale=scale, coefficient=coefficient, mlp=mlp.eval(),
                statistical_optimizer_updates=updates,
                fitted_readout_parameters=int(coefficient.numel()+sum(p.numel() for p in mlp.parameters())))


def predict_readouts(fitted, features):
    output = {'linear': [], 'mlp': []}
    with torch.inference_mode():
        for left in range(0, len(features), 8192):
            x = torch.as_tensor(np.array(features[left:left+8192], dtype=np.float32))
            x = ((x-fitted['mean'])/fitted['scale']).float()
            a = torch.cat((torch.ones(len(x), 1), x), 1).double()
            output['linear'].append((a @ fitted['coefficient']).numpy())
            output['mlp'].append(fitted['mlp'](x).squeeze(-1).double().numpy())
    return {key: np.concatenate(values) for key, values in output.items()}


def calibrated_mask(calibration_scores, scores, requested_fraction):
    """Frozen TRAIN threshold applied independently to every held-out row.

    Global top-k on held-out scores would let a later prefix change an earlier
    routing decision. We deliberately report realized, not nominal, fractions.
    """
    calibration_scores, scores = np.asarray(calibration_scores), np.asarray(scores)
    if (not 0 <= requested_fraction <= 1 or not len(calibration_scores)
            or not np.isfinite(calibration_scores).all() or not np.isfinite(scores).all()):
        raise ValueError('Finite calibration/prefix scores and fraction required')
    if requested_fraction in (0., 1.):
        return np.full(len(scores), bool(requested_fraction)), None
    threshold = float(np.quantile(calibration_scores, 1-requested_fraction))
    return scores > threshold, threshold


def allocation_curve(calibration, scores, losses, domains,
                     fractions=(0., .1, .25, .5, .75, .9, 1.)):
    """Offline output selection only; no claim about routed KV trajectories."""
    losses = np.asarray(losses)
    if losses.shape != (2, len(scores)) or not np.isfinite(losses).all():
        raise ValueError('Aligned recorded depth-one/two losses required')
    weight = domain_weights(domains)
    shallow, deep = losses @ weight
    gain = losses[0]-losses[1]
    rows = []
    for q in fractions:
        chosen, threshold = calibrated_mask(calibration, scores, q)
        fraction = float(chosen.mean())
        nll = float(np.dot(weight, np.where(chosen, losses[1], losses[0])))
        random_nll = float(shallow-fraction*np.dot(weight, gain))
        rows.append(dict(requested_train_fraction=q, realized_token_fraction=fraction,
            threshold=threshold, selected_targets=int(chosen.sum()), nll=nll,
            expected_same_fraction_random_nll=random_nll,
            gain_over_same_fraction_random=random_nll-nll,
            gain_over_all_depth_two=float(deep-nll),
            domain_nll={name: float(np.where(chosen, losses[1], losses[0])[np.asarray(domains) == key].mean())
                        for name, key in [('web', 1), ('math', 2), ('code', 3)]}))
    return rows


def matched_curve_nll(curve, fraction):
    """Descriptive interpolation by realized cost only, never by target loss.

    Interpolation weights use held-out feature frequencies. This is a
    retrospective expected-cost comparison, not a fresh calibrated policy.
    """
    rows = sorted(curve, key=lambda x: x['realized_token_fraction'])
    for a, b in zip(rows, rows[1:]):
        qa, qb = a['realized_token_fraction'], b['realized_token_fraction']
        if qa <= fraction <= qb and qb > qa:
            mix = (fraction-qa)/(qb-qa)
            return float((1-mix)*a['nll']+mix*b['nll'])
    raise ValueError('Curve must bracket the requested realized cost')
