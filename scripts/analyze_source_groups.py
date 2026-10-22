"""Report source-group uncertainty separately from initialization uncertainty."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from compute_trainium.metrics import (DocumentScore, paired_group_interval,
                                      paired_three_seed_interval, summarize)


def mean_scores(pools):
    """Average losses across seeds while keeping each held-out target once."""
    keyed = [{s.document: s for s in pool} for pool in pools]
    if not keyed or any(len(k) != len(p) for k, p in zip(keyed, pools)):
        raise ValueError("Empty or repeated document scores")
    first = keyed[0]
    if any(k.keys() != first.keys() for k in keyed):
        raise ValueError("Seeds scored different documents")
    result = []
    for identifier, row in first.items():
        if any((k[identifier].domain, k[identifier].group, k[identifier].tokens) !=
               (row.domain, row.group, row.tokens) for k in keyed):
            raise ValueError("Seeds scored different targets or source groups")
        result.append(replace(row, nll_sum=sum(k[identifier].nll_sum for k in keyed)/len(keyed)))
    return result


def analyze(experiment, root, draws=2000):
    records = json.loads((experiment/"progress.json").read_text())["records"]
    output = []
    for step in sorted({r["update"] for r in records}):
        arms = {a: {r["seed"]: r for r in records if r["arm"] == a and r["update"] == step}
                for a in ("ordinary", "early")}
        seeds = sorted(arms["ordinary"].keys() & arms["early"].keys())
        if not seeds:
            continue
        pools = {a: [] for a in arms}
        within_seed = []
        domain_gains = {d: {} for d in ("web", "math", "code")}
        for seed in seeds:
            for arm in arms:
                record = arms[arm][seed]
                path = root/"runs"/(record["run"]+"-dev")/"document-scores.json"
                pool = [DocumentScore(**s) for s in json.loads(path.read_text())]
                metrics = summarize(pool)
                if abs(metrics["weighted_nll"]-record["metrics"]["weighted_nll"]) > 1e-10:
                    raise ValueError("Document scores differ from frozen curve report")
                pools[arm].append(pool)
            a, b = pools["ordinary"][-1], pools["early"][-1]
            within_seed.append({"seed": seed, **paired_group_interval(a, b, draws=draws, seed=23)})
            for domain in domain_gains:
                domain_gains[domain][seed] = (summarize(a)["domains"][domain]["nll"]-
                                             summarize(b)["domains"][domain]["nll"])
        reference, candidate = (mean_scores(pools[a]) for a in ("ordinary", "early"))
        mean_interval = paired_group_interval(reference, candidate, draws=draws, seed=23)
        domains = {}
        for domain, gains in domain_gains.items():
            domains[domain] = {"seed_gains_nats": gains,
                "source_group_interval_for_mean_model_gain": paired_group_interval(
                    [s for s in reference if s.domain == domain],
                    [s for s in candidate if s.domain == domain],
                    draws=draws, seed=23, weights={domain: 1.})}
            if len(gains) == 3:
                domains[domain]["paired_initialization_interval"] = paired_three_seed_interval(gains)
        output.append({"update": step, "paired_seeds": seeds, "within_seed_source_group_intervals": within_seed,
                       "source_group_interval_for_mean_model_gain": mean_interval, "domains": domains})
    report = {"cohort": "curve", "test_sealed": True, "bootstrap_draws": draws, "bootstrap_seed": 23,
              "milestones": output,
              "interpretation": "Descriptive source-group bootstrap, conditional on fitted models. Separate from the primary paired initialization interval; domain comparisons are exploratory."}
    destination = experiment/"analysis/source-group-uncertainty.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--draws", type=int, default=2000)
    args = p.parse_args()
    analyze(args.experiment, args.root, args.draws)
