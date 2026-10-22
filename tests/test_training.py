import copy

import pytest
import torch

from compute_trainium.training import (PackedCorpus, TokenObjective, WSDSchedule,
                                       atomic_checkpoint, global_indices)
from tests.test_core import tiny


def docs():
    return [{"id": f"{d}{i}", "domain": d, "split": "train", "tokens": [4+i]*(i+1)}
            for d in ("web", "math", "code") for i in range(7)]


def test_packing_covers_short_docs_tails_and_counted_boundaries_once():
    packed = PackedCorpus(docs(), context=8)
    b, labels = packed.batch(list(range(packed.windows)), pauses=2)
    expected = sorted(t for d in docs() for t in d["tokens"])
    assert sorted(b.targets[b.target_mask].tolist()) == expected
    assert int(b.target_mask.sum()) == packed.real_targets == len(expected)
    assert b.target_mask.equal(labels > 0)
    assert packed.counts(0, packed.windows) == [28]*3
    assert packed.counts(0, packed.windows*3, repeat=True) == [84]*3
    assert packed.counts(packed.windows-1, 3, repeat=True) == [
        int((packed.batch([packed.windows-1, 0, 1])[1] == d).sum()) for d in (1, 2, 3)]
    with pytest.raises(ValueError):
        packed.batch([packed.windows])
    assert packed.fingerprint == PackedCorpus(docs()[::-1], 8).fingerprint


def test_global_stream_cursor_restart_is_arm_independent():
    flat = [i for micro in range(3) for rank in range(4)
            for i in global_indices(24, micro, rank, 4, 2)]
    assert flat == list(range(24, 48))


def test_variable_mask_accumulation_matches_full_batch_gradient():
    packed = PackedCorpus(docs(), 8)
    b, _ = packed.batch([0, 1, 2])
    a, z = tiny(), tiny()
    obj_a, obj_z = TokenObjective(a, b.prediction_positions, device="cpu"), TokenObjective(z, b.prediction_positions, device="cpu")
    denominator = int(b.target_mask.sum())
    (obj_a(b.inputs, b.targets, b.target_mask)/denominator).backward()
    for i in range(3):
        (obj_z(b.inputs[i:i+1], b.targets[i:i+1], b.target_mask[i:i+1])/denominator).backward()
    for p, q in zip(a.parameters(), z.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)


def test_wsd_has_complete_annealing_at_each_branch_endpoint():
    trunk = WSDSchedule(1e-3, 100)
    branch = WSDSchedule(1e-3, 100, 800, 1000)
    assert trunk.lr(10000) == branch.lr(800) == 1e-3
    assert branch.lr(50) == 5e-4
    assert branch.lr(1000) == branch.lr(1200) == pytest.approx(1e-4)
    assert branch.lr(900) == pytest.approx(5.5e-4)
    with pytest.raises(ValueError):
        WSDSchedule(1e-3, 100, 50, 1000)


def test_full_atomic_checkpoint_preserves_optimizer_and_stream_cursor(tmp_path):
    model = tiny()
    opt = torch.optim.AdamW(model.parameters())
    model(torch.tensor([[1, 4, 5]])).square().mean().backward()
    opt.step()
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "cursor": 48,
             "tokens": 123, "rng": torch.get_rng_state()}
    atomic_checkpoint(tmp_path/"state.pt", state)
    saved = torch.load(tmp_path/"state.pt", weights_only=False)
    from scripts.qualify_native import digest
    assert digest(state) == digest(saved)
    assert not (tmp_path/"state.tmp").exists()


def test_long_timing_budget_anneals_all_flop_endpoints_and_charges_branches():
    from scripts.performance_budget import measured_budget
    ordinary = {"status": "PASS_PACKED_TRAINING_PHASE", "warm_updates": 100,
                "world_size": 64, "parameters": 150_436_608, "context": 512,
                "local_batch": 1, "accumulation": 8, "precision": "bf16", "optimizer": "plain",
                "pauses": 0, "warm_real_targets": 25_600_000,
                "warm_global_real_targets_per_second": 500_000,
                "flops_per_global_step": {"training_matmul_flops": 256_000_000}}
    pause = copy.deepcopy(ordinary)
    pause.update(pauses=16, flops_per_global_step={"training_matmul_flops": 320_000_000})
    report = measured_budget(ordinary, pause, tokens=1_000_000)
    assert report["annealed_equal_flop_endpoints"]["early"]["total_real_targets"] == 800_000
    assert report["annealed_equal_flop_endpoints"]["late"]["total_real_targets"] == 900_000
    assert all(x["decay_complete_at_comparison"] for x in report["annealed_equal_flop_endpoints"].values())
    assert report["additional_flop_decay_branches_steady_hours"] > 0
    pause["warm_updates"] = 12
    with pytest.raises(ValueError):
        measured_budget(ordinary, pause)


def test_supervisor_recognizes_silent_compile_failure_log():
    from scripts.supervise import FATAL
    assert FATAL.search("[Neuron] [ERROR] [CompilationCache.cpp:962] Compilation error key=x error= Command failed (neuronx-cc compilation) with exit code 70")
    assert not FATAL.search("WARN NRT: using threaded asynchronous implementation")
    assert not FATAL.search("attn_backend='sdpa' cannot express per-document masking")
