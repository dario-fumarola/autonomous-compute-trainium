"""Real-token likelihood and paired source-group uncertainty."""

import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch


def real_token_losses(logits, targets, target_mask):
    # Explicit full-vocabulary CE avoids a native beta cross_entropy graph break.
    values = logits.float()
    losses = values.logsumexp(dim=-1) - values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
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


def paired_three_seed_interval(gain_by_seed):
    """Student t interval across the three predeclared paired training seeds."""
    if len(gain_by_seed) != 3 or not all(math.isfinite(x) for x in gain_by_seed.values()):
        raise ValueError("Pass three finite paired seed gains.")
    gains = list(gain_by_seed.values())
    mean = statistics.mean(gains)
    half = 4.302652729911275 * statistics.stdev(gains) / math.sqrt(3)
    return {"mean_gain_nats": mean, "interval_95": [mean-half, mean+half],
            "paired_seed_gains": gain_by_seed, "assumption": "Independent paired seeds with approximately normal gains; df=2."}
