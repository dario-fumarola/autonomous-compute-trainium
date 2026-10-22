"""Real-token likelihood and paired source-group uncertainty."""

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


def real_token_losses(logits, targets, target_mask):
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), reduction="none").reshape_as(targets)
    return losses * target_mask


@dataclass
class DocumentScore:
    domain: str
    document: str
    group: str
    nll_sum: float
    tokens: int


def summarize(scores, weights=None):
    weights = weights or {"web": .6, "math": .2, "code": .2}
    if not scores:
        raise ValueError("No document scores.")
    domains = defaultdict(lambda: {"nll_sum": 0., "real_targets": 0, "documents": 0})
    seen = set()
    for s in scores:
        if s.tokens <= 0 or not math.isfinite(s.nll_sum) or s.document in seen:
            raise ValueError("Invalid or repeated document score.")
        seen.add(s.document)
        d = domains[s.domain]
        d["nll_sum"] += s.nll_sum
        d["real_targets"] += s.tokens
        d["documents"] += 1
    if set(domains) != set(weights) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Domain coverage or mixture weights do not match.")
    for d in domains.values():
        d["nll"] = d["nll_sum"] / d["real_targets"]
    return {"domains": dict(domains),
            "weighted_nll": sum(weights[k] * domains[k]["nll"] for k in weights),
            "weights": weights, "unit": "nats per real source token"}


def paired_group_interval(reference, candidate, *, draws=2000, seed=0, weights=None):
    """Positive means candidate improves NLL; resample groups jointly across domains."""
    a, b = ({s.document: s for s in rows} for rows in (reference, candidate))
    if len(a) != len(reference) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError("Pairing needs identical unique documents.")
    for k in a:
        if (a[k].domain, a[k].group, a[k].tokens) != (b[k].domain, b[k].group, b[k].tokens):
            raise ValueError("Paired targets or source groups differ.")
    weights = weights or {"web": .6, "math": .2, "code": .2}
    summarize(reference, weights)
    summarize(candidate, weights)
    groups = sorted({s.group for s in reference})
    if len(groups) < 2 or draws < 2:
        raise ValueError("At least two source groups and draws are required.")
    grouped = defaultdict(list)
    for k in a:
        grouped[a[k].group].append(k)
    rng, values = random.Random(seed), []
    for _ in range(draws):
        sums = defaultdict(lambda: [0., 0])
        for g in rng.choices(groups, k=len(groups)):
            for k in grouped[g]:
                sums[a[k].domain][0] += a[k].nll_sum - b[k].nll_sum
                sums[a[k].domain][1] += a[k].tokens
        if any(sums[d][1] == 0 for d in weights):
            continue
        values.append(sum(weights[d] * sums[d][0] / sums[d][1] for d in weights))
    if len(values) < .9 * draws:
        raise ValueError("Too few source groups per domain for a stable bootstrap.")
    values.sort()
    gain = summarize(reference, weights)["weighted_nll"] - summarize(candidate, weights)["weighted_nll"]
    return {"gain_nats": gain, "interval_95": [values[int(.025 * len(values))],
            values[min(len(values) - 1, int(.975 * len(values)))]],
            "source_groups": len(groups), "accepted_draws": len(values),
            "warning": "Within-seed uncertainty only; report paired training seeds separately."}
