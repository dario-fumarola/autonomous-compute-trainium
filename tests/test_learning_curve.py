import json

import pytest

from compute_trainium.metrics import DocumentScore
from scripts.analyze_learning_curve import analyze
from scripts.analyze_source_groups import mean_scores


def test_curve_pairs_seeds_and_exposures_and_preserves_gain_direction(tmp_path):
    records = []
    for seed in (23, 47, 71):
        for arm in ("ordinary", "early"):
            value = 2.0 if arm == "ordinary" else 1.97
            records.append({"arm": arm, "seed": seed, "update": 384,
                            "real_tokens": 100_000_000, "training_matmul_flops": 10,
                            "metrics": {"weighted_nll": value,
                                        "domains": {d: {"nll": value} for d in ("web", "math", "code")}}})
    (tmp_path/"progress.json").write_text(json.dumps({"records": records}))
    result = analyze(tmp_path)
    assert result["paired"][0]["mean_gain_nats"] == pytest.approx(.03)
    assert result["paired"][0]["assessment_at_this_exposure"] == "useful_gain_supported"
    assert (tmp_path/"analysis/learning-curve.svg").exists()
    records[-1]["real_tokens"] += 1
    (tmp_path/"progress.json").write_text(json.dumps({"records": records}))
    with pytest.raises(AssertionError, match="exposures"):
        analyze(tmp_path)


def test_mean_seed_scores_keep_targets_once_and_reject_changed_groups():
    first = [DocumentScore("web", "fixture", "group-a", 12., 4)]
    second = [DocumentScore("web", "fixture", "group-a", 8., 4)]
    mean = mean_scores([first, second])[0]
    assert mean.nll_sum == 10.
    assert mean.tokens == 4
    second[0].group = "group-b"
    with pytest.raises(ValueError, match="source groups"):
        mean_scores([first, second])


def test_receipt_collection_counts_failures_and_ends_at_worker_closure(tmp_path):
    from scripts.collect_learning_experiment import collect
    root, experiment, output = (tmp_path/p for p in ("root", "experiment", "public"))
    for p in (root/"snapshots/campaign-failed", experiment, output):
        p.mkdir(parents=True)
    (root/"snapshots/campaign-failed/launch.json").write_text(json.dumps({"started_unix": 200}))
    (root/"snapshots/campaign-failed/closure.json").write_text(json.dumps(
        {"exit_code": 1, "wall_seconds": 100, "assigned_chip_hours": .4}))
    (experiment/"specification.json").write_text(json.dumps({"identity": "campaign"}))
    (experiment/"complete.json").write_text("{}")
    (output/"corpus-manifest.json").write_text(json.dumps({"preparation_wall_seconds": 30}))
    result = collect(root, experiment, output, "1970-01-01T00:01:40+00:00")
    assert result["assigned_chip_hours_including_failures"] == .4
    assert result["inclusive_node_hours"] == pytest.approx(200/3600)
    assert result["runs"][0]["status"] == "FAILED_OR_INCOMPLETE"
