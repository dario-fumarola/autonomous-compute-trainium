"""One bounded training phase. New shapes and DEV run in fresh processes.

Qualification may explicitly repeat the small natural corpus for timing. A
scientific invocation never wraps the data cursor and requires a frozen corpus.
"""

import argparse
import contextlib
import copy
import ctypes
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

_cores = os.environ["RESEARCH_VISIBLE_CORES"].split(",")
os.environ["NEURON_RT_VISIBLE_CORES"] = _cores[int(os.environ.get("LOCAL_RANK", "0"))]

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from compute_trainium.model import ModelConfig
from compute_trainium.data import pad_layout
from compute_trainium.titan import build_titan
from compute_trainium.training import (PackedCorpus, TokenObjective, WSDSchedule,
                                       atomic_checkpoint, file_sha256, global_indices)
from scripts.qualify_native import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--packed-corpus", type=Path)
    p.add_argument("--dev-corpus", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", choices=["tiny", "small"], default="small")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--pauses", type=int, choices=[0, 16], default=0)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--steps", type=int, default=104, help="Updates in this process, not total run")
    p.add_argument("--warmup", type=int, default=4, help="Unmeasured compile/warm updates")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--data-seed", type=int)
    p.add_argument("--pad-to", type=int, default=0)
    p.add_argument("--reference-threads", type=int, default=8)
    p.add_argument("--purpose", choices=["qualification", "tuning", "scientific"], default="qualification")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warmup-tokens", type=int, default=0)
    p.add_argument("--decay-start", type=int)
    p.add_argument("--decay-end", type=int)
    p.add_argument("--decay-branch", action="store_true")
    p.add_argument("--clip", type=float, default=1.)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--qualification-repeat", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--check-global", action="store_true")
    p.add_argument("--optimizer", choices=["plain", "foreach"], default="plain")
    args = p.parse_args()
    if args.steps <= args.warmup or min(args.batch, args.accumulation, args.clip) <= 0:
        raise ValueError("Positive sizes and a nonempty warm interval required")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(2)
    torch.neuron.set_device(0)
    dist.init_process_group("neuron")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    config = (ModelConfig(dim=128, layers=2, heads=4, kv_heads=2, ffn_dim=256, max_seq=2048)
              if args.size == "tiny" else ModelConfig())
    corpus_hash = file_sha256(args.corpus)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    packed = (PackedCorpus.from_mapped(args.packed_corpus, args.context, data_seed, corpus_hash)
              if args.packed_corpus else PackedCorpus(json.loads(args.corpus.read_text()), args.context, data_seed))
    bf16 = args.precision == "bf16"
    torch.manual_seed(args.seed)
    model = build_titan(config)
    initial_model_digest = digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  foreach=args.optimizer == "foreach", fused=False)
    schedule = WSDSchedule(args.lr, args.lr_warmup_tokens, args.decay_start, args.decay_end)
    signature = {"model": asdict(config), "context": args.context, "batch": args.batch,
                 "accumulation": args.accumulation, "world": world, "seed": args.seed,
                 "precision": args.precision, "corpus_sha256": corpus_hash,
                 "packing_sha256": packed.fingerprint, "clip": args.clip,
                 "optimizer": args.optimizer, "repeat_qualification": args.qualification_repeat}
    if args.dev_corpus:
        signature["dev_corpus_sha256"] = file_sha256(args.dev_corpus)
    if args.data_seed is not None:
        signature["data_seed"] = data_seed
    if args.pad_to:
        signature["pad_to"] = args.pad_to
    cursor = exposure = update = cumulative_flops = 0
    previous_pauses, recovered_digest = None, None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint configuration/data fingerprint mismatch")
        # Schedule overrides are allowed only for an explicitly separate decay branch.
        if saved["schedule"] != asdict(schedule):
            if not args.decay_branch or args.decay_start != saved["real_tokens"]:
                raise ValueError("New decay must explicitly branch at the recovered stable exposure")
            old = dict(saved["schedule"], decay_start=None, decay_end=None)
            new = dict(asdict(schedule), decay_start=None, decay_end=None)
            if old != new or saved["schedule"]["decay_start"] is not None:
                raise ValueError("A branch must preserve peak/warmup and fork an undecayed trunk")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        cursor, exposure, update = saved["cursor"], saved["real_tokens"], saved["update"]
        cumulative_flops = saved["training_matmul_flops"]
        previous_pauses = saved["pauses"]
        recovered_digest = digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        if recovered_digest != saved["state_digest"]:
            raise AssertionError("Full CPU model/optimizer recovery differs")
        if saved.get("initial_model_digest", initial_model_digest) != initial_model_digest:
            raise AssertionError("Random initialization differs from the original fit")
        del saved
    origin_exposure, origin_update = exposure, update
    def batch_at(micro, worker=rank):
        batch = packed.batch(global_indices(cursor, micro, worker, world, args.batch),
                             args.pauses, repeat=args.qualification_repeat)[0]
        return pad_layout(batch, args.pad_to)
    first = batch_at(0)
    # A single graph is compiled in this process. Validate BEFORE DDP or updates.
    reference = copy.deepcopy(model) if rank == 0 else None
    # Moving the model must precede moving the recovered optimizer tensors.
    model.to("neuron")
    recovered_optimizer = optimizer.state_dict() if args.resume else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  foreach=args.optimizer == "foreach", fused=False)
    if recovered_optimizer is not None:
        optimizer.load_state_dict(recovered_optimizer)
        del recovered_optimizer
        if digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()}) != recovered_digest:
            raise AssertionError("Full recovery changed when transferred to the accelerator")
    if any(v.dtype != torch.float32 for v in model.parameters()):
        raise AssertionError("Master parameters must remain FP32")
    # Full checkpoint recovery can leave >1 GiB/rank in malloc arenas. Return
    # those before the pause compiler's peak; file-backed checkpoint pages are
    # shared across ranks rather than copied by torch.load.
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    objective = TokenObjective(model, first.prediction_positions, bf16)
    torch._dynamo.config.optimize_ddp = False
    compiled = torch.compile(objective, backend="neuron", fullgraph=True, dynamic=False)
    x, y, m = (v.contiguous().to("neuron") for v in (first.inputs, first.targets, first.target_mask))
    probe = compiled(x, y, m)/max(1, first.target_mask.sum().item())
    probe.backward()
    cpu_check = None
    if rank == 0:
        torch.set_num_threads(args.reference_threads)
        ref = TokenObjective(reference, first.prediction_positions, bf16, "cpu")
        ref_loss = ref(first.inputs, first.targets, first.target_mask)/first.target_mask.sum()
        ref_loss.backward()
        error = total = 0.
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            error += (a.grad.detach().cpu()-b.grad).square().sum().item()
            total += b.grad.square().sum().item()
        relative = (error/max(total, 1e-30))**.5
        native_loss, cpu_loss = probe.detach().cpu().item(), ref_loss.item()
        if not torch.isfinite(torch.tensor(native_loss)) or abs(native_loss-cpu_loss) > (.03 if bf16 else .003):
            raise AssertionError(f"Compile probe loss differs: {native_loss} vs {cpu_loss}")
        if not relative <= (.08 if bf16 else .02):
            raise AssertionError(f"Compile probe full gradient differs: {relative}")
        cpu_check = {"native_loss": native_loss, "cpu_loss": cpu_loss,
                     "gradient_relative_l2": relative, "bf16_reference": bf16}
        (args.output/"cpu-probe.json").write_text(json.dumps(cpu_check, indent=2)+"\n")
        print(json.dumps({"event": "cpu_probe_pass", **cpu_check}), flush=True)
        torch.set_num_threads(2)
    del probe
    optimizer.zero_grad(set_to_none=True)
    if reference is not None:
        reference.zero_grad(set_to_none=True)
    ddp = DDP(compiled, device_ids=None, broadcast_buffers=False, bucket_cap_mb=64,
              gradient_as_bucket_view=True)
    rows, warm_tokens, cold_seconds = [], 0, None
    domain_counts = [0, 0, 0]
    global_check = None
    last_checkpoint_update = None
    def checkpoint():
        nonlocal last_checkpoint_update
        dist.barrier()
        last_checkpoint_update = update
        if rank == 0:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            state_digest = digest(state)
            atomic_checkpoint(args.output/"checkpoint.pt", {**state, "state_digest": state_digest,
                "signature": signature, "schedule": asdict(schedule), "pauses": args.pauses,
                "initial_model_digest": initial_model_digest,
                "cursor": cursor, "real_tokens": exposure, "update": update,
                "training_matmul_flops": cumulative_flops, "cpu_rng": torch.get_rng_state(),
                "numerical_probe_passed": True})
            print(json.dumps({"event": "checkpoint", "update": update, "real_tokens": exposure}), flush=True)
        dist.barrier()
    dist.barrier()
    torch.neuron.synchronize()
    warm_started = None
    for step in range(args.steps):
        if step == args.warmup:
            gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
            dist.barrier()
            torch.neuron.synchronize()
            warm_started = time.monotonic()
        before = time.monotonic()
        # Count masks globally on CPU from the immutable stream, without a device
        # collective or mean-of-means bias. DDP averages gradients across ranks.
        counts = packed.counts(cursor, world*args.batch*args.accumulation,
                               repeat=args.qualification_repeat)
        global_count = sum(counts)
        if global_count <= 0:
            raise AssertionError("No real targets in global update")
        optimizer.zero_grad(set_to_none=True)
        lr = schedule.lr(exposure+global_count)
        for group in optimizer.param_groups:
            group["lr"] = lr
        local_sum = torch.zeros((), device="neuron")
        for micro in range(args.accumulation):
            b = batch_at(micro)
            x, y, m = (v.contiguous().to("neuron") for v in (b.inputs, b.targets, b.target_mask))
            with ddp.no_sync() if micro < args.accumulation-1 else contextlib.nullcontext():
                loss_sum = ddp(x, y, m)
                (loss_sum*(world/global_count)).backward()
            local_sum += loss_sum.detach()
        if step == 0 and args.check_global and rank == 0:
            reference.zero_grad(set_to_none=True)
            ref = TokenObjective(reference, first.prediction_positions, bf16, "cpu")
            for micro in range(args.accumulation):
                for worker in range(world):
                    b = batch_at(micro, worker)
                    (ref(b.inputs, b.targets, b.target_mask)/global_count).backward()
            error = total = 0.
            for a, b in zip(model.parameters(), reference.parameters(), strict=True):
                error += (a.grad.detach().cpu()-b.grad).square().sum().item()
                total += b.grad.square().sum().item()
            relative = (error/max(total, 1e-30))**.5
            if not relative <= (.08 if bf16 else .02):
                raise AssertionError(f"Accumulated global gradient mismatch: {relative}")
            global_check = {"gradient_relative_l2": relative, "real_targets": global_count}
        # Check and clip the global averaged FP32 gradient before AdamW.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, foreach=False)
        norm_value = norm.detach().cpu().item()
        if not torch.isfinite(torch.tensor(norm_value)):
            raise AssertionError("Nonfinite global gradient norm")
        optimizer.step()
        if step == 0:
            for state in optimizer.state.values():
                if any(state[name].dtype != torch.float32 for name in ("exp_avg", "exp_avg_sq")):
                    raise AssertionError("Adam moments must stay FP32 under autocast")
        cursor += world*args.batch*args.accumulation
        exposure += global_count
        update += 1
        cumulative_flops += config.flops(args.context, first.inputs.shape[1],
                                         world*args.batch*args.accumulation)["training_matmul_flops"]
        domain_counts = [a+b for a, b in zip(domain_counts, counts)]
        if step >= args.warmup:
            warm_tokens += global_count
        if step == 0 or (step+1) % 10 == 0 or step == args.steps-1:
            value = local_sum.detach().cpu().item()/global_count*world
            if not torch.isfinite(torch.tensor(value)):
                raise AssertionError("Nonfinite loss")
            row = {"update": update, "real_tokens": exposure, "lr": lr,
                   "rank0_scaled_nll": value, "gradient_norm": norm_value,
                   "elapsed_seconds": time.monotonic()-start}
            rows.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
        if step == 0:
            cold_seconds = time.monotonic()-before
            # Reference is no longer needed after optional global validation.
            del reference
        if args.checkpoint_every and update % args.checkpoint_every == 0:
            checkpoint()
    torch.neuron.synchronize()
    dist.barrier()
    warm_seconds = time.monotonic()-warm_started
    if args.save and last_checkpoint_update != update:
        checkpoint()
    state_digest = digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    (args.output/f"replica-{rank}.json").write_text(json.dumps({"rank": rank, "state_digest": state_digest})+"\n")
    dist.barrier()
    if rank == 0:
        if len({json.loads((args.output/f"replica-{r}.json").read_text())["state_digest"] for r in range(world)}) != 1:
            raise AssertionError("Replica full states differ")
        import torch_neuronx
        report = {"status": "PASS_PACKED_TRAINING_PHASE", "scientific_result": False,
                  "purpose": args.purpose, "initial_model_digest": initial_model_digest,
                  "reference_threads": args.reference_threads,
                  "size": args.size, "parameters": config.parameter_count(),
                  "world_size": world, "physical_chips": sorted({int(c)//4 for c in _cores}),
                  "context": args.context, "local_batch": args.batch,
                  "accumulation": args.accumulation, "precision": args.precision,
                  "input_layout": "host_contiguous",
                  "data_backend": "mapped" if args.packed_corpus else "in_memory_qualification",
                  "optimizer": args.optimizer, "pauses": args.pauses,
                  "updates": update-origin_update, "warm_updates": args.steps-args.warmup,
                  "warm_seconds": warm_seconds, "warm_real_targets": warm_tokens,
                  "warm_global_real_targets_per_second": warm_tokens/warm_seconds,
                  "phase_real_targets": exposure-origin_exposure, "total_real_targets": exposure,
                  "cursor": cursor, "corpus_unique_train_targets": packed.real_targets,
                  "qualification_repeats_data": args.qualification_repeat,
                  "domain_real_targets": dict(zip(("web", "math", "code"), domain_counts)),
                  "cpu_probe": cpu_check, "global_gradient_check": global_check,
                  "all_replica_model_optimizer_states_equal": True,
                  "recovered_model_optimizer_digest": recovered_digest,
                  "previous_pauses": previous_pauses, "state_digest": state_digest,
                  "flops_per_global_step": config.flops(args.context, first.inputs.shape[1],
                                                         world*args.batch*args.accumulation),
                  "total_training_matmul_flops": cumulative_flops,
                  "cold_first_training_step_seconds": cold_seconds,
                  "cold_probe_and_setup_seconds": rows[0]["elapsed_seconds"]-cold_seconds,
                  "wall_seconds": time.monotonic()-start,
                  "fallback_ops_rank0": str(torch_neuronx.get_fallback_ops()),
                  "signature": signature, "schedule": asdict(schedule), "progress": rows,
                  "scope": "Packed natural data, H2D, token-weighted no_sync accumulation, FP32 masters/AdamW, clipping and WSD. Per-step gradient finiteness synchronizes. Final state save/hash excluded; periodic saves included."}
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
