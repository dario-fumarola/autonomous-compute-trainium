"""Price fully annealed token/FLOP endpoints from long packed-trainer timings."""

import argparse
import json
from pathlib import Path


def measured_budget(ordinary, pause, tokens=3_000_000_000, seeds=3, decay_fraction=.1):
    for report in (ordinary, pause):
        if report["status"] != "PASS_PACKED_TRAINING_PHASE" or report["warm_updates"] < 100:
            raise ValueError("Need passing packed-trainer timings with >=100 warm updates")
    for key in ("world_size", "parameters", "context", "local_batch", "accumulation", "precision", "optimizer", "input_layout"):
        if ordinary[key] != pause[key]:
            raise ValueError(f"Mismatched timing condition: {key}")
    if ordinary["pauses"] != 0 or pause["pauses"] != 16:
        raise ValueError("Need both mechanisms")
    if not 0 < decay_fraction < .5:
        raise ValueError("Declare a short decay fraction")
    speeds = [r["warm_global_real_targets_per_second"] for r in (ordinary, pause)]
    costs = [r["flops_per_global_step"]["training_matmul_flops"]*r["warm_updates"]/r["warm_real_targets"]
             for r in (ordinary, pause)]
    flop_budget = tokens*costs[0]
    switch = tokens/2
    endpoints = {"ordinary": tokens, "early": flop_budget/costs[1],
                 "late": switch+(flop_budget-switch*costs[0])/costs[1]}
    def hours(arm, target):
        before = min(target, switch) if arm == "late" else target if arm == "ordinary" else 0
        return (before/speeds[0]+(target-before)/speeds[1])/3600
    token_hours = {arm: hours(arm, tokens) for arm in endpoints}
    branches = {arm: {"total_real_targets": int(end),
                       "stable_trunk_targets": int((1-decay_fraction)*end),
                       "decay_targets": int(decay_fraction*end),
                       "decay_complete_at_comparison": True}
                for arm, end in endpoints.items()}
    # Ordinary token and FLOP endpoints coincide. Other FLOP endpoints require
    # additional decay branches; the stable prefix is shared with the trunk.
    branch_extra = seeds*sum(decay_fraction*endpoints[arm]/speeds[1]/3600 for arm in ("early", "late"))
    training = seeds*sum(token_hours.values())
    # Mini LR fits expose both mechanisms in the late arm at their own midpoint.
    tuning = 3*100_000_000*(1/speeds[0]+1/speeds[1]+.5/speeds[0]+.5/speeds[1])/3600
    total = training+branch_extra+tuning
    return {"status": "PROPOSED_NOT_LAUNCHED", "parameters": ordinary["parameters"],
            "real_tokens_per_token_matched_fit": tokens, "paired_seeds": seeds,
            "tokens_per_parameter": tokens/ordinary["parameters"],
            "chosen_batch": {k: ordinary[k] for k in ("world_size", "context", "local_batch", "accumulation", "precision", "optimizer", "input_layout")},
            "measured_real_tps": dict(zip(("ordinary", "pause"), speeds)),
            "estimated_training_matmul_flops_per_real_target": dict(zip(("ordinary", "pause"), costs)),
            "equal_flop_budget_per_fit": flop_budget, "late_switch_real_targets": int(switch),
            "annealed_equal_flop_endpoints": branches,
            "token_matched_steady_hours_per_seed": token_hours,
            "token_matched_steady_hours_all_seeds": training,
            "additional_flop_decay_branches_steady_hours": branch_extra,
            "lr_sweep_steady_hours": tuning, "steady_total_hours": total,
            "planning_node_hours_2x": 2*total, "planning_node_hours_3x": 3*total,
            "additional_paired_3B_to_30B_steady_hours": (30_000_000_000-tokens)*sum(1/s for s in speeds)/3600,
            "limits": ["No large run is authorized by this estimate. Freeze corpus/order, LR, schedule and launch gates first.",
                       "Every endpoint includes its decay inside the stated budget; all extra branches are charged.",
                       "Round branch/switch endpoints to complete shared global updates; record realized tokens and FLOPs.",
                       "2x-3x allowance includes unmeasured production data IO, periodic eval/checkpoint/restarts and variation.",
                       "LR sweeps must use this batch/precision; three variants, three candidates, 100M real targets each.",
                       "Matmul FLOPs include vocabulary head but exclude elementwise/optimizer work; not a hardware counter.",
                       "Qualification repeats a small natural corpus; its learning curve is not a scientific result."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ordinary", type=Path, required=True)
    p.add_argument("--pause", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = measured_budget(json.loads(args.ordinary.read_text()), json.loads(args.pause.read_text()))
    args.output.write_text(json.dumps(result, indent=2)+"\n")
