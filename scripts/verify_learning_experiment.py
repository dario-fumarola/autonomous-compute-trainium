"""Audit the realized comparison against its frozen specification and receipts."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def audit(experiment, root, data, require_complete=False):
    specification = read(experiment/"specification.json")
    manifest = read(data/"manifest.json")
    assert sha(data/"manifest.json") == specification["data_manifest_sha256"]
    assert sha(Path("docs/LEARNING_CURVE_PROTOCOL.md")) == specification["protocol_sha256"]
    assert sha(data/"dev.json") == manifest["dev_sha256"]
    assert sha(data/"test-sealed.json") == manifest["test_sha256"]
    assert manifest["source_group_overlap"] == 0
    selected = read(experiment/"selected-lrs.json")
    assert selected["selection_cohort"] == "tune"
    assert selected["curve_evaluations_before_selection"] == 0
    assert selected["specification_sha256"] == sha(experiment/"specification.json")
    trials = read(experiment/"tuning.json")
    assert len(trials) == 6
    source = read(root/"snapshots"/trials[0]["run"]/"launch.json")["source_sha256"]
    stable_code = ("scripts/train_distributed.py", "scripts/evaluate_checkpoint.py",
                   "compute_trainium/training.py", "compute_trainium/titan.py",
                   "compute_trainium/model.py", "compute_trainium/metrics.py", "compute_trainium/data.py")
    for arm in specification["arms"]:
        candidates = [t for t in trials if t["arm"] == arm and t["status"] == "PASS"]
        expected = min(candidates, key=lambda t: (t["tune_weighted_nll"], t["lr"]))["lr"]
        assert selected["selected_lrs"][arm] == expected
    progress = read(experiment/"progress.json") if (experiment/"progress.json").exists() else {"records": []}
    records = progress["records"]
    complete = (experiment/"complete.json").exists()
    if require_complete:
        assert complete and len(records) == 18
        assert {(r["seed"], r["arm"], r["update"]) for r in records} == {
            (s, a, u) for s in specification["initialization_seeds"]
            for a in specification["arms"] for u in specification["main_update_milestones"]}
    initial_digests, checks = {}, []
    for item in [*trials, *records]:
        run = item["run"]
        if item.get("status") == "FAILED":
            assert item["failure_class"] == "nonfinite_learning_rate_candidate"
            continue
        trained = read(root/"runs"/run/"report.json")
        evaluated = read(root/"runs"/(run+"-dev")/"report.json")
        is_main = item in records
        assert trained["parameters"] == 150436608 and trained["world_size"] == 64
        assert trained["all_replica_model_optimizer_states_equal"]
        assert not trained["qualification_repeats_data"]
        signature = trained["signature"]
        assert signature["data_seed"] == specification["data_seed"]
        assert signature["packing_sha256"] == manifest["packing"]["fingerprint"]
        assert signature.get("pad_to", 0) == specification["arms"][item["arm"]]["pad_to"]
        assert trained["pauses"] == specification["arms"][item["arm"]]["pauses"]
        assert evaluated["split"] == "dev" and evaluated["cohort"] == ("curve" if is_main else "tune")
        assert evaluated["weights_unchanged"] and evaluated["max_documents_per_domain"] == 0
        seed = signature["seed"]
        initial_digests.setdefault(seed, set()).add(trained["initial_model_digest"])
        if is_main:
            assert trained["purpose"] == "scientific"
            assert trained["total_real_targets"] == specification["boundaries_real_targets"][str(item["update"])]
            assert trained["total_training_matmul_flops"] == trained["flops_per_global_step"]["training_matmul_flops"]*item["update"]
            assert evaluated["metrics"] == item["metrics"]
            assert (root/"runs"/(run+"-dev")/"report.json").stat().st_mtime >= selected["frozen_unix"]
            if item["update"] == 3840:
                assert math.isclose(trained["progress"][-1]["lr"], selected["selected_lrs"][item["arm"]]*.1)
            if item["update"] > 384:
                assert trained["recovered_model_optimizer_digest"]
        for phase in (run, run+"-dev"):
            launch = read(root/"snapshots"/phase/"launch.json")
            assert all(launch["source_sha256"][name] == source[name] for name in stable_code)
            closure = read(root/"snapshots"/phase/"closure.json")
            supervisor = read(root/"logs"/(phase+".supervisor.json"))
            assert closure["exit_code"] == 0 and supervisor["worker_exit_code"] == 0
            assert supervisor["supervisor_failure"] is None and supervisor["container_oom_kill_delta"] == 0
        checks.append({"run": run, "cohort": evaluated["cohort"],
                       "real_targets": trained["total_real_targets"],
                       "absolute_cpu_native_probe_nll_difference": abs(trained["cpu_probe"]["native_loss"]-trained["cpu_probe"]["cpu_loss"]),
                       "full_gradient_relative_l2": trained["cpu_probe"]["gradient_relative_l2"],
                       "absolute_evaluation_cpu_native_probe_nll_difference": abs(evaluated["cpu_probe"]["native_nll"]-evaluated["cpu_probe"]["cpu_nll"])})
    assert all(len(digests) == 1 for digests in initial_digests.values())
    assert len({next(iter(v)) for v in initial_digests.values()}) == len(initial_digests)
    result = {"status": "PASS_COMPLETE_PROTOCOL_AUDIT" if require_complete else "PASS_COMPLETED_RECORDS_AUDIT",
              "complete": complete, "tuning_trials": len(trials), "curve_records": len(records),
              "test_file_unchanged": True, "data_and_protocol_hashes_match": True,
              "matched_initializations_and_data": True, "frozen_lrs_match_tuning_minima": True,
              "identical_training_model_loss_and_evaluation_code": {name: source[name] for name in stable_code},
              "checks": checks}
    (experiment/"protocol-audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    audit(args.experiment, args.root, args.data, args.require_complete)
