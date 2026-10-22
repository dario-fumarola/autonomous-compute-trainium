import copy

import pytest

from scripts.budget import budget


def test_budget_requires_matched_measurements_and_accounts_for_all_arms():
    a = {"status": "PASS_NATIVE_DDP_BENCHMARK", "size": "small", "world_size": 64,
         "parameters": 150_436_608, "context": 512, "local_batch": 1,
         "physical_chips": list(range(16)), "pauses": 0,
         "warm_global_real_targets_per_second": 100_000,
         "flops_per_global_step": {"training_matmul_flops": 100}}
    b = copy.deepcopy(a)
    b.update(pauses=16, warm_global_real_targets_per_second=80_000,
             flops_per_global_step={"training_matmul_flops": 125})
    r = budget(a, b, tokens=1_000_000_000)
    assert r["fits"] == 9
    assert r["flop_matched_real_target_milestones"]["early"] == 800_000_000
    assert r["flop_matched_real_target_milestones"]["late"] == 900_000_000
    assert r["planning_hours_2x"] > 2*r["steady_training_hours_all_seeds"]
    b["parameters"] += 1
    with pytest.raises(ValueError):
        budget(a, b)
