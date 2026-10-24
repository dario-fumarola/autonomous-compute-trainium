"""Hindsight document-depth selection, exclusively an exploratory upper bound."""
import numpy as np


def weighted_document_losses(scores, weights):
    totals = {d: sum(s['tokens'] for s in scores if s['domain'] == d) for d in weights}
    if any(n <= 0 for n in totals.values()) or any(s['domain'] not in weights for s in scores):
        raise ValueError('Missing or unexpected domain')
    if any(s['tokens'] <= 0 or not np.isfinite(s['nll_sum']) or s['nll_sum'] < 0 for s in scores):
        raise ValueError('Invalid real-target score')
    return np.array([weights[s['domain']]*s['nll_sum']/totals[s['domain']] for s in scores])


def hindsight_point(losses, costs, penalty):
    """Minimize the separable sum of weighted NLL plus normalized work.

    Arrays are depth×document. Each column must describe the SAME source
    document at every depth. Ties prefer the earlier (cheaper) depth.
    """
    losses, costs = np.asarray(losses), np.asarray(costs)
    if (losses.ndim != 2 or losses.shape != costs.shape or not losses.size
            or not np.isfinite(losses).all() or not np.isfinite(costs).all()
            or (losses < 0).any() or (costs <= 0).any()
            or not np.isfinite(penalty) or penalty < 0):
        raise ValueError('Invalid oracle objective')
    choice = np.argmin(losses+penalty*costs, axis=0)
    columns = np.arange(losses.shape[1])
    return {'choice': choice, 'weighted_nll': float(losses[choice, columns].sum()),
            'normalized_inference_work': float(costs[choice, columns].sum())}
