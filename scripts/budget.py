"""Budget a gated experiment from matching measured whole-instance reports."""

import argparse
import json
import math
from pathlib import Path


def budget(ordinary, pause, *, tokens=3_000_000_000, seeds=3):
    for d in (ordinary, pause):
        if d["status"] != "PASS_NATIVE_DDP_BENCHMARK" or d["size"] != "small":
            raise ValueError("Budget requires passing smallest-size hardware reports.")
    for key in ("world_size", "parameters", "context", "local_batch", "physical_chips"):
        if ordinary[key] != pause[key]:
            raise ValueError(f"Benchmark conditions differ: {key}.")
    if ordinary["pauses"] != 0 or pause["pauses"] != 16:
        raise ValueError("Need ordinary and 16-pause measurements.")
    normal_tps = ordinary["warm_global_real_targets_per_second"]
    pause_tps = pause["warm_global_real_targets_per_second"]
    normal_flops = ordinary["flops_per_global_step"]["training_matmul_flops"]
    pause_flops = pause["flops_per_global_step"]["training_matmul_flops"]
    ratio = pause_flops / normal_flops
    one_seed_hours = {"ordinary": tokens/normal_tps/3600,
                      "early": tokens/pause_tps/3600,
                      "late_estimate": (tokens/2/normal_tps + tokens/2/pause_tps)/3600}
    steady_hours = seeds * sum(one_seed_hours.values())
    # A separate tuning allowance; its candidates are not scientific replications.
    tuning_tokens_per_candidate = 100_000_000
    tuning_hours = 3 * sum(tuning_tokens_per_candidate/tokens*x for x in one_seed_hours.values())
    global_batch = ordinary["world_size"] * ordinary["local_batch"] * ordinary["context"]
    return {"status": "PROPOSED_NOT_LAUNCHED", "parameters": ordinary["parameters"],
            "tokens_per_fit_requested": tokens, "seeds": seeds, "fits": 3*seeds,
            "global_real_targets_per_full_step": global_batch,
            "full_steps_per_fit": math.ceil(tokens/global_batch),
            "tokens_per_parameter": tokens/ordinary["parameters"],
            "measured_tps": {"ordinary": normal_tps, "pause": pause_tps},
            "estimated_pause_training_flop_ratio": ratio,
            "flop_matched_real_target_milestones": {"ordinary": tokens,
                "early": int(tokens/ratio), "late": int(tokens/2 + tokens/2/ratio)},
            "milestone_scope": "Curve comparisons on a common source-token schedule; independently optimized FLOP-budget endpoints require additional runs.",
            "steady_training_hours_per_seed": one_seed_hours,
            "steady_training_hours_all_seeds": steady_hours,
            "lr_tuning": {"candidates_per_variant": 3, "real_tokens_per_candidate": tuning_tokens_per_candidate,
                          "steady_hours": tuning_hours},
            "planning_hours_2x": 2*(steady_hours+tuning_hours),
            "planning_hours_3x": 3*(steady_hours+tuning_hours),
            "planning_chip_hours_2x": 2*(steady_hours+tuning_hours)*len(ordinary["physical_chips"]),
            "planning_chip_hours_3x": 3*(steady_hours+tuning_hours)*len(ordinary["physical_chips"]),
            "extra_paired_continuation_3B_to_30B_steady_hours": (30_000_000_000-tokens)*(1/normal_tps+1/pause_tps)/3600,
            "continuation_scope": "One ordinary/early paired seed only; late-training learning diagnostic, not three-seed convergence evidence.",
            "limits": ["Warm timing is short; 2x-3x allowance includes an unmeasured data/eval/checkpoint reserve, not a throughput guarantee.",
                       "Only FP32 and the measured shape/topology are priced; no assumed BF16 speedup or MFU.",
                       "Prepare a larger immutable natural corpus and near-duplicate audit before training; the qualification sample is too small.",
                       "Three billion targets is a signal-seeking pilot, not a convergence claim.",
                       "This FLOP estimate excludes elementwise and optimizer work. Add recomputation if activation checkpointing is enabled."]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ordinary", type=Path, required=True)
    p.add_argument("--pause", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = budget(json.loads(args.ordinary.read_text()), json.loads(args.pause.read_text()))
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
