"""Account for model calls and actual worker/node time without double counting."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(root, experiment, receipts):
    ledger = read(receipts/"resource-ledger.json")
    costs = defaultdict(lambda: defaultdict(int))
    timing = defaultdict(lambda: defaultdict(float))
    incomplete = []
    warm = defaultdict(lambda: [0, 0.])
    windows_by_cohort = defaultdict(set)
    for row in ledger["runs"]:
        run = row["run"]
        category = "qualification" if run.startswith("learning-pad") else "tuning" if "-lr-" in run else "main"
        evaluation = run.endswith("-dev")
        timing[category]["evaluation_worker_wall_seconds" if evaluation else "training_worker_wall_seconds"] += row["closure"].get("wall_seconds", 0)
        report_path = receipts/(run+"-report.json")
        if not report_path.exists() or row["status"] == "FAILED_OR_INCOMPLETE":
            incomplete.append(run)
            continue
        report = read(report_path)
        if evaluation:
            trained = read(receipts/(run.removesuffix("-dev")+"-report.json"))
            slots = trained["world_size"]*trained["local_batch"]*trained["accumulation"]
            per_window = trained["flops_per_global_step"]["forward_matmul_flops"]//slots
            assert per_window*slots == trained["flops_per_global_step"]["forward_matmul_flops"]
            scores = read(root/"runs"/run/"document-scores.json")
            assert sum(s["tokens"] for s in scores) == report["targets"]
            windows = sum((s["tokens"]+trained["context"]-1)//trained["context"] for s in scores)
            windows_by_cohort[report["cohort"]].add(windows)
            costs[category]["native_dev_forward_matmul_flops"] += per_window*windows
            costs[category]["cpu_dev_probe_matmul_flops"] += per_window
            costs[category]["scored_dev_real_targets"] += report["targets"]
        else:
            forward = report["flops_per_global_step"]["forward_matmul_flops"]
            training = report["flops_per_global_step"]["training_matmul_flops"]
            assert training == 3*forward
            assert report["global_gradient_check"] is None, "Additional global probes need separate accounting"
            costs[category]["optimizer_training_matmul_flops"] += training*report["updates"]
            costs[category]["optimizer_training_real_targets"] += report["phase_real_targets"]
            # Every rank runs one local-batch forward/backward probe, before
            # accumulation. Rank zero additionally runs that batch on CPU.
            costs[category]["native_training_probe_matmul_flops"] += training//report["accumulation"]
            costs[category]["cpu_training_probe_matmul_flops"] += training//(report["world_size"]*report["accumulation"])
            if category == "main":
                arm = "early" if report["pauses"] else "ordinary"
                warm[arm][0] += report["warm_real_targets"]
                warm[arm][1] += report["warm_seconds"]
    assert all(len(values) == 1 for values in windows_by_cohort.values())
    progress = read(experiment/"progress.json")
    endpoints = [r for r in progress["records"] if r["update"] == 3840]
    if len(endpoints) == 6:
        assert costs["main"]["optimizer_training_matmul_flops"] == sum(r["training_matmul_flops"] for r in endpoints)
    result = {"status": "COMPLETE" if (experiment/"complete.json").exists() and not incomplete else "INTERIM",
              "dense_model_call_estimates": dict(costs), "measured_worker_wall_seconds": dict(timing),
              "inclusive_node_hours": ledger["inclusive_node_hours"],
              "assigned_chip_hours_including_failures": ledger["assigned_chip_hours_including_failures"],
              "preparation_wall_seconds_overlaps_node_window": ledger["data_preparation_wall_seconds"],
              "main_warm_real_targets_per_second": {a: n/t for a, (n, t) in warm.items()},
              "windows_per_evaluation": {c: next(iter(v)) for c, v in windows_by_cohort.items()},
              "runs_without_complete_flop_accounting": incomplete,
              "assumptions": "Dense full-square attention; FMA=2, backward=2x forward, full LM head and tail padding included. Model-call estimates, not hardware counters. Excludes optimizer/elementwise operations, compilation and software-test fixtures from FLOPs; actual node/worker time includes those costs. CPU and native numerical probes are listed separately. Failed worker time is retained even when its FLOPs are unknown. Do not add overlapping preparation and node times."}
    (experiment/"cost-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True)
    args = p.parse_args()
    summarize(args.root, args.experiment, args.receipts)
