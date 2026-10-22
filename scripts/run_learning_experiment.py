"""Freeze tuning choices, then produce the paired natural-text learning curves."""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--identity", default="natural-curve-v1")
    args = p.parse_args()
    root = Path(os.environ.get("RESEARCH_ROOT", "/run/user/1000/autonomous-compute"))
    args.output.mkdir(parents=True, exist_ok=False)
    data_manifest = json.loads((args.data/"manifest.json").read_text())
    if data_manifest["status"] != "PASS_FROZEN_NATURAL_CORPUS" or data_manifest["source_group_overlap"]:
        raise ValueError("Data launch gate incomplete")
    pad_run = "learning-pad1536-qualification-001"
    pad = json.loads((root/"runs"/pad_run/"report.json").read_text())
    clean = json.loads((root/"logs"/(pad_run+".supervisor.json")).read_text())
    if (pad["warm_updates"] < 100 or pad["status"] != "PASS_PACKED_TRAINING_PHASE"
            or clean["worker_exit_code"] or clean["supervisor_failure"]
            or not pad["all_replica_model_optimizer_states_equal"]):
        raise ValueError("Padding qualification gate failed")
    if pad["warm_global_real_targets_per_second"] <= 353878.73335139017:
        raise ValueError("Padding did not improve measured throughput")
    prefix = np.load(args.data/"packed/prefix_counts.npy", mmap_mode="r")
    # One optimizer update consumes 64 ranks x batch 2 x accumulation 2 windows.
    def targets(step):
        return int(prefix[step*256].sum())
    boundaries = {str(s): targets(s) for s in (16, 40, 346, 384, 1152, 3456, 3840)}
    data_relative = args.data.relative_to(root)
    container_data = "/work/"+str(data_relative)
    common = ["--size", "small", "--context", "1024", "--batch", "2", "--accumulation", "2",
              "--precision", "bf16", "--data-seed", "23", "--reference-threads", "8",
              "--corpus", container_data+"/source.json", "--packed-corpus", container_data+"/packed",
              "--dev-corpus", container_data+"/dev.json", "--warmup", "4"]
    started = time.time()
    specification = {"identity": args.identity, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "protocol_sha256": sha(Path("docs/LEARNING_CURVE_PROTOCOL.md")),
                     "data_manifest_sha256": sha(args.data/"manifest.json"),
                     "boundaries_real_targets": boundaries, "initialization_seeds": [23, 47, 71],
                     "data_seed": 23, "arms": {"ordinary": {"pauses": 0, "pad_to": 0},
                                               "early": {"pauses": 16, "pad_to": 1536}},
                     "tuning_lrs": [3e-4, 1e-3, 3e-3], "tuning_seed": 11,
                     "primary_cohort": "curve", "test_sealed": True,
                     "claim_threshold_nats": .01, "main_update_milestones": [384, 1152, 3840]}
    (args.output/"specification.json").write_text(json.dumps(specification, indent=2)+"\n")
    records = []
    def phase_plan(plan, name):
        plan_file = args.output/(name+"-plan.json")
        plan_file.write_text(json.dumps(plan, indent=2)+"\n")
        subprocess.run(["python3", "-m", "scripts.run_phases", "--plan", str(plan_file),
                        "--output", str(args.output/(name+"-controller"))], check=True)
    def progress():
        (args.output/"progress.json").write_text(json.dumps({"specification": specification,
            "records": records, "elapsed_seconds": time.time()-started}, indent=2)+"\n")
    trials = []
    for arm, mechanism in specification["arms"].items():
        for index, lr in enumerate(specification["tuning_lrs"]):
            run = f"{args.identity}-lr-{arm}-{index}"
            arguments = [*common, "--purpose", "tuning", "--seed", "11", "--lr", str(lr),
                         "--lr-warmup-tokens", str(targets(16)), "--decay-start", str(targets(346)),
                         "--decay-end", str(targets(384))]
            if mechanism["pad_to"]:
                arguments += ["--pad-to", str(mechanism["pad_to"])]
            plan = {"qualification_only": True, "world": 64, "training_memory_gib": 256,
                    "evaluation_memory_gib": 64, "timeout_seconds": 1800,
                    "evaluation_cohort": "tune", "training_arguments": arguments,
                    "phases": [{"run": run, "steps": 384, "pauses": mechanism["pauses"]}]}
            try:
                phase_plan(plan, run)
                report = json.loads((root/"runs"/(run+"-dev")/"report.json").read_text())
                trial = {"arm": arm, "lr": lr, "run": run, "status": "PASS",
                         "tune_weighted_nll": report["metrics"]["weighted_nll"]}
            except subprocess.CalledProcessError as error:
                trial = {"arm": arm, "lr": lr, "run": run, "status": "FAILED", "returncode": error.returncode}
                # A compiler/resource failure is not evidence a learning rate is
                # poor. Stop instead of excluding such a trial from selection.
                supervisor_path = root/"logs"/(run+".supervisor.json")
                log_path = root/"logs"/(run+".log")
                supervisor = json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
                numerical_failure = (supervisor.get("supervisor_failure") is None and log_path.exists()
                                     and "Nonfinite global gradient norm" in log_path.read_text())
                if numerical_failure:
                    trial["failure_class"] = "nonfinite_learning_rate_candidate"
                    trials.append(trial)
                    (args.output/"tuning.json").write_text(json.dumps(trials, indent=2)+"\n")
                    continue
                trials.append(trial)
                (args.output/"tuning.json").write_text(json.dumps(trials, indent=2)+"\n")
                raise
            trials.append(trial)
            (args.output/"tuning.json").write_text(json.dumps(trials, indent=2)+"\n")
            print(json.dumps({"event": "tuning_trial_complete", **trial}), flush=True)
    selected = {arm: min((r for r in trials if r["arm"] == arm and r["status"] == "PASS"), key=lambda r: (r["tune_weighted_nll"], r["lr"]))["lr"]
                for arm in specification["arms"]}
    frozen = {"selected_lrs": selected, "selection_cohort": "tune", "curve_evaluations_before_selection": 0,
              "frozen_unix": time.time(), "specification_sha256": sha(args.output/"specification.json")}
    (args.output/"selected-lrs.json").write_text(json.dumps(frozen, indent=2)+"\n")
    ordinary_tps, early_tps = 855335.9846585225, pad["warm_global_real_targets_per_second"]
    main_hours = 3*targets(3840)*(1/ordinary_tps+1/early_tps)/3600
    tuning_hours = 3*targets(384)*(1/ordinary_tps+1/early_tps)/3600
    launch = {"immutable_sources": True, "near_duplicate_audit": data_manifest["near_duplicate_audit"],
              "frozen_order_sha256": data_manifest["packing"]["order_fingerprint"],
              "lr_selected_at_this_batch": frozen, "trainer_qualified": True,
              "measured_budget": {"scope": "Two arms, three seeds, 1B targets; 8 node-hour cap including tuning/evaluation",
                                  "steady_main_hours": main_hours, "steady_tuning_hours": tuning_hours,
                                  "planning_node_hours_2x": 2*(main_hours+tuning_hours),
                                  "planning_node_hours_3x": 3*(main_hours+tuning_hours),
                                  "node_hours_limit": 8, "ordinary_tps_reference": 855335.9846585225,
                                  "pause_tps_measured": pad["warm_global_real_targets_per_second"]}}
    manifest_path = args.output/"launch-manifest.json"
    manifest_path.write_text(json.dumps(launch, indent=2)+"\n")
    # Complete each token milestone for all paired seeds before extending any
    # model. This produces usable, aligned early curves without changing the plan.
    previous = {}
    for step, count in ((384, 384), (1152, 768), (3840, 2688)):
        for seed in specification["initialization_seeds"]:
            for arm, mechanism in specification["arms"].items():
                if time.time()-started > 8*3600:
                    raise RuntimeError("Predeclared eight-hour experiment cap reached")
                run = f"{args.identity}-{arm}-s{seed}-u{step}"
                arguments = [*common, "--purpose", "scientific", "--seed", str(seed), "--lr", str(selected[arm]),
                             "--lr-warmup-tokens", str(targets(40)), "--decay-start", str(targets(3456)),
                             "--decay-end", str(targets(3840)), "--checkpoint-every", "384"]
                if mechanism["pad_to"]:
                    arguments += ["--pad-to", str(mechanism["pad_to"])]
                plan = {"qualification_only": False, "scientific_manifest": str(manifest_path),
                        "world": 64, "training_memory_gib": 256, "evaluation_memory_gib": 64,
                        "timeout_seconds": 3600, "evaluation_cohort": "curve", "training_arguments": arguments,
                        "phases": [{"run": run, "steps": count, "pauses": mechanism["pauses"]}]}
                if (seed, arm) in previous:
                    plan["resume"] = previous[(seed, arm)]
                phase_plan(plan, run)
                previous[(seed, arm)] = f"/work/runs/{run}/checkpoint.pt"
                trained = json.loads((root/"runs"/run/"report.json").read_text())
                evaluated = json.loads((root/"runs"/(run+"-dev")/"report.json").read_text())
                record = {"run": run, "arm": arm, "seed": seed, "update": step,
                          "real_tokens": trained["total_real_targets"], "metrics": evaluated["metrics"],
                          "initial_model_digest": trained["initial_model_digest"],
                          "training_matmul_flops": trained["total_training_matmul_flops"]}
                if record["real_tokens"] != targets(step) or evaluated["cohort"] != "curve":
                    raise AssertionError("Exposure or evaluation cohort differs from the frozen plan")
                for other in records:
                    if other["seed"] == seed and other["initial_model_digest"] != record["initial_model_digest"]:
                        raise AssertionError("Paired random initializations differ")
                records.append(record)
                progress()
                subprocess.run(["/home/ubuntu/ema-trainium/.venv/bin/python", "-m", "scripts.analyze_learning_curve",
                                "--experiment", str(args.output)], check=True)
                print(json.dumps({"event": "curve_checkpoint_complete", "run": run,
                                  "real_tokens": record["real_tokens"], "nll": record["metrics"]["weighted_nll"]}), flush=True)
    (args.output/"complete.json").write_text(json.dumps({"status": "COMPLETE", "elapsed_seconds": time.time()-started})+"\n")


if __name__ == "__main__":
    main()
