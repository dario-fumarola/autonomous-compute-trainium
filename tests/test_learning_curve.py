import json

import pytest

from scripts.analyze_learning_curve import analyze


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
