"""Static-per-rank multi-depth DDP, separately qualified from frozen studies.

Based on train_fast.py at d3442e3. All arms share deterministic within-update
window routing and the qualified unscaled fused AdamW implementation.
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
from dataclasses import asdict, replace
from pathlib import Path

_cores = os.environ["RESEARCH_VISIBLE_CORES"].split(",")
os.environ["NEURON_RT_VISIBLE_CORES"] = _cores[int(os.environ.get("LOCAL_RANK", "0"))]

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from compute_trainium.model import ModelConfig
from compute_trainium.data import pad_layout
from compute_trainium.titan import build_titan
from compute_trainium.recurrence import RecurrenceConfig, build_from_config
from compute_trainium.preln_recurrence import PreNormRecurrenceConfig
from compute_trainium.continuation import validate_resume_signature, validate_resume_schedule
from compute_trainium.training import (PackedCorpus, TokenObjective, WSDSchedule,
                                       atomic_checkpoint, file_sha256, global_indices)
from scripts.qualify_native import digest
from compute_trainium.depth_mixture import rank_depths, routed_indices, mixture_step_flops
from compute_trainium.pretraining_optimizer import make_optimizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--packed-corpus", type=Path)
    p.add_argument("--dev-corpus", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", choices=["tiny", "small"], default="small")
    p.add_argument("--architecture", choices=["ordinary", "recurrent", "preln"], default="ordinary")
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--untied-depth", action="store_true")
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
    p.add_argument("--extend-stable-schedule", action="store_true")
    p.add_argument("--allow-corpus-extension", action="store_true")
    p.add_argument("--clip", type=float, default=1.)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--qualification-repeat", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--check-global", action="store_true")
    p.add_argument("--optimizer", choices=["plain", "foreach", "fused"], default="plain")
    p.add_argument("--warm-rendezvous", type=Path)
    p.add_argument("--warm-peer-names", default="")
    p.add_argument("--depth-policy", choices=["ordinary", "fixed", "mixture"], required=True)
    p.add_argument("--routing-seed", type=int, default=271828)
    p.add_argument("--recipe", choices=["pretraining"], default="pretraining")
    args = p.parse_args()
    if args.steps <= args.warmup or min(args.batch, args.accumulation, args.clip) <= 0:
        raise ValueError("Positive sizes and a nonempty warm interval required")
    if args.warm_rendezvous:
        peers=args.warm_peer_names.split(',')
        if len(set(peers))!=2 or args.output.name not in peers or any(Path(v).name!=v for v in peers):
            raise ValueError('Concurrent timing requires the two explicit unique job identities')
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(2)
    torch.neuron.set_device(0)
    dist.init_process_group("neuron")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    config = (ModelConfig(dim=128, layers=2, heads=4, kv_heads=2, ffn_dim=256, max_seq=2048)
              if args.size == "tiny" else ModelConfig())
    if args.architecture in ("recurrent", "preln"):
        cls = PreNormRecurrenceConfig if args.architecture == "preln" else RecurrenceConfig
        config = (cls(dim=128, layers=4, heads=4, kv_heads=2, ffn_dim=256,
                                  max_seq=2048, prelude=1, core=2, coda=1,
                                  repetitions=args.repetitions, untied=args.untied_depth)
                  if args.size == "tiny" else
                  cls(repetitions=args.repetitions, untied=args.untied_depth))
    if args.optimizer not in ['plain', 'fused']:
        raise ValueError('Explicit recipe permits plain or qualified fused AdamW only')
    if args.architecture != 'preln' or args.untied_depth or args.pauses or args.pad_to:
        raise ValueError('This experiment changes only training-depth distribution')
    depths = rank_depths(world, args.depth_policy)
    config = replace(config, repetitions=1 if args.depth_policy == 'ordinary' else 2)
    nominal_config = config
    local_config = replace(config, repetitions=depths[rank])
    corpus_hash = file_sha256(args.corpus)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    packed = (PackedCorpus.from_mapped(args.packed_corpus, args.context, data_seed, corpus_hash)
              if args.packed_corpus else PackedCorpus(json.loads(args.corpus.read_text()), args.context, data_seed))
    bf16 = args.precision == "bf16"
    torch.manual_seed(args.seed)
    model = build_from_config(local_config)
    initial_model_digest = digest(model.state_dict())
    optimizer = make_optimizer(model, args.lr, fused=args.optimizer == "fused")
    schedule = WSDSchedule(args.lr, args.lr_warmup_tokens, args.decay_start, args.decay_end)
    signature = {"model": asdict(config), "context": args.context, "batch": args.batch,
                 "accumulation": args.accumulation, "world": world, "seed": args.seed,
                 "precision": args.precision, "corpus_sha256": corpus_hash,
                 "packing_sha256": packed.fingerprint, "clip": args.clip,
                 "optimizer": args.optimizer, "repeat_qualification": args.qualification_repeat}
    signature.update(depth_policy=args.depth_policy, rank_depths=list(depths),
                     routing_seed=args.routing_seed, routing='splitmix64_fisher_yates_v1',
                     optimizer_betas=[.9,.95], matrix_weight_decay=.1, embedding_norm_weight_decay=0., eps=1e-8)
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
        metadata = json.loads((args.packed_corpus/'manifest.json').read_text()) if args.packed_corpus else None
        validate_resume_signature(saved['signature'], signature,
            allow_extension=args.allow_corpus_extension, metadata=metadata, packed=packed,
            cursor=saved['cursor'], exposure=saved['real_tokens'])
        validate_resume_schedule(saved['schedule'], asdict(schedule), saved['real_tokens'],
            decay_branch=args.decay_branch, extend_stable=args.extend_stable_schedule)
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
        batch = packed.batch(routed_indices(cursor, micro, worker, world, args.batch, args.accumulation, args.routing_seed),
                             args.pauses, repeat=args.qualification_repeat)[0]
        return pad_layout(batch, args.pad_to)
    first = batch_at(0)
    # A single graph is compiled in this process. Validate BEFORE DDP or updates.
    probe_ranks = [depths.index(d) for d in sorted(set(depths))]
    reference = copy.deepcopy(model) if rank in probe_ranks else None
    # Moving the model must precede moving the recovered optimizer tensors.
    model.to("neuron")
    recovered_optimizer = optimizer.state_dict() if args.resume else None
    optimizer = make_optimizer(model, args.lr, fused=args.optimizer == "fused")
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
    if rank in probe_ranks:
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
        (args.output/f"cpu-probe-rank{rank}-depth{depths[rank]}.json").write_text(json.dumps(cpu_check, indent=2)+"\n")
        print(json.dumps({"event": "cpu_probe_pass", **cpu_check}), flush=True)
        torch.set_num_threads(2)
    del probe
    optimizer.zero_grad(set_to_none=True)
    if reference is not None:
        reference.zero_grad(set_to_none=True)
    ddp = DDP(compiled, device_ids=None, broadcast_buffers=False, bucket_cap_mb=64,
              gradient_as_bucket_view=True)
    step_flops = (mixture_step_flops(nominal_config, world, args.batch, args.accumulation, args.context)
                  if args.depth_policy == 'mixture' else
                  nominal_config.flops(args.context, args.context, world*args.batch*args.accumulation)['training_matmul_flops'])
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
                "numerical_probe_passed": True, "depth_policy": args.depth_policy})
            print(json.dumps({"event": "checkpoint", "update": update, "real_tokens": exposure}), flush=True)
        dist.barrier()
    dist.barrier()
    torch.neuron.synchronize()
    warm_started = None
    warm_started_unix = None
    for step in range(args.steps):
        if step == args.warmup:
            gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
            dist.barrier()
            torch.neuron.synchronize()
            if args.warm_rendezvous:
                if rank==0:
                    args.warm_rendezvous.mkdir(parents=True,exist_ok=True)
                    ready=args.warm_rendezvous/(args.output.name+'.json')
                    if ready.exists():raise FileExistsError('Timing rendezvous must be fresh')
                    temporary=ready.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'member':args.output.name,'peers':peers,'ready_unix':time.time()}))
                    temporary.replace(ready);limit=time.monotonic()+300
                    while not all((args.warm_rendezvous/(v+'.json')).exists() for v in peers):
                        if time.monotonic()>limit:raise TimeoutError('Other half-node job did not complete compilation')
                        time.sleep(.1)
                    for v in peers:
                        if json.loads((args.warm_rendezvous/(v+'.json')).read_text())['peers']!=peers:
                            raise ValueError('Timing peer identities differ')
                dist.barrier()
            warm_started = time.monotonic()
            warm_started_unix = time.time()
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
            gradients = [torch.zeros_like(v) for v in reference.parameters()]
            for depth in sorted(set(depths)):
                depth_model = build_from_config(replace(nominal_config, repetitions=depth))
                depth_model.load_state_dict(reference.state_dict())
                ref = TokenObjective(depth_model, first.prediction_positions, bf16, "cpu")
                for micro in range(args.accumulation):
                    for worker in range(world):
                        if depths[worker] != depth:
                            continue
                        b = batch_at(micro, worker)
                        (ref(b.inputs, b.targets, b.target_mask)/global_count).backward()
                for total_gradient, parameter in zip(gradients, depth_model.parameters(), strict=True):
                    total_gradient.add_(parameter.grad)
                del depth_model, ref
            error = total = 0.
            for a, b in zip(model.parameters(), gradients, strict=True):
                error += (a.grad.detach().cpu()-b).square().sum().item()
                total += b.square().sum().item()
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
        cumulative_flops += step_flops
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
    warm_ended_unix = time.time()
    if args.save and last_checkpoint_update != update:
        checkpoint()
    state_digest = digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    import torch_neuronx
    (args.output/f"replica-{rank}.json").write_text(json.dumps({"rank": rank, "state_digest": state_digest,
        "initial_model_digest": initial_model_digest, "depth": depths[rank],
        "fallback_ops": list(torch_neuronx.get_fallback_ops())})+"\n")
    dist.barrier()
    if rank == 0:
        if len({json.loads((args.output/f"replica-{r}.json").read_text())["state_digest"] for r in range(world)}) != 1:
            raise AssertionError("Replica full states differ")
        replicas=[json.loads((args.output/f"replica-{r}.json").read_text()) for r in range(world)]
        if len({v['initial_model_digest'] for v in replicas})!=1:
            raise AssertionError('Mixed-depth replicas have different initial parameters')
        if any(set(v['fallback_ops']) - {'aten::_to_copy'} for v in replicas):
            raise AssertionError('Unexpected native fallback in a training-depth path')
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
                  "warm_started_unix": warm_started_unix, "warm_ended_unix": warm_ended_unix,
                  "warm_global_real_targets_per_second": warm_tokens/warm_seconds,
                  "phase_real_targets": exposure-origin_exposure, "total_real_targets": exposure,
                  "cursor": cursor, "corpus_unique_train_targets": packed.real_targets,
                  "qualification_repeats_data": args.qualification_repeat,
                  "domain_real_targets": dict(zip(("web", "math", "code"), domain_counts)),
                  "cpu_probe": cpu_check, "global_gradient_check": global_check,
                  "all_replica_model_optimizer_states_equal": True,
                  "recovered_model_optimizer_digest": recovered_digest,
                  "previous_pauses": previous_pauses, "state_digest": state_digest,
                  "flops_per_global_step": {"training_matmul_flops": step_flops},
                  "depth_policy": args.depth_policy, "rank_depths": list(depths),
                  "mean_training_passes": sum(depths)/world,
                  "local_repetitions_rank0": depths[0],
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
