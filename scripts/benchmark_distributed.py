"""Bounded native DDP throughput with real source batches on all visible cores."""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

# Give each rank one disjoint logical core before the backend loads.
_cores = os.environ["RESEARCH_VISIBLE_CORES"].split(",")
os.environ["NEURON_RT_VISIBLE_CORES"] = _cores[int(os.environ["LOCAL_RANK"])]

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from compute_trainium.data import document_windows, layout
from compute_trainium.model import ModelConfig
from compute_trainium.titan import build_titan
from scripts.qualify_native import Objective, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", choices=["tiny", "small"], default="tiny")
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--pauses", type=int, choices=[0, 16], default=0)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--check-cpu-global", action="store_true")
    p.add_argument("--inference", action="store_true")
    args = p.parse_args()
    if args.steps <= args.warmup or args.warmup < 2:
        raise ValueError("Need at least two warmup and one measured updates.")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(2)
    device = "neuron"
    torch.neuron.set_device(0)
    dist.init_process_group("neuron")
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(dim=128, layers=2, heads=4, kv_heads=2, ffn_dim=256,
                         max_seq=1024) if args.size == "tiny" else ModelConfig(**json.loads(
                             (Path(__file__).resolve().parents[1]/"configs/models.json").read_text())["small"])
    corpus = json.loads(args.corpus.read_text())
    pools = {}
    for domain in ("web", "math", "code"):
        # Full natural windows keep every rank's target denominator identical.
        pools[domain] = [s for d in corpus if d["split"] == "train" and d["domain"] == domain
                         for s, m in document_windows(d["tokens"], args.context) if m.all()]
        random.Random("23:" + domain).shuffle(pools[domain])
    cycle = ["web"]*6 + ["math"]*2 + ["code"]*2
    def batch_at(step, worker):
        examples = []
        for j in range(args.batch):
            index = (step*world + worker)*args.batch + j
            domain = cycle[index % 10]
            ordinal = (index//10)*cycle.count(domain) + cycle[:index % 10].count(domain)
            if ordinal >= len(pools[domain]):
                raise ValueError("Bounded benchmark exhausted its unique natural windows.")
            examples.append(pools[domain][ordinal])
        return layout(torch.stack(examples), torch.ones(args.batch, args.context, dtype=torch.bool),
                      pauses=args.pauses)
    torch.manual_seed(23)
    model = build_titan(config)
    cpu_reference = copy.deepcopy(model) if rank == 0 and args.check_cpu_global else None
    model.to(device)
    first = batch_at(0, rank)
    objective = Objective(model, first.prediction_positions.to(device))
    torch._dynamo.config.optimize_ddp = False
    compiled = torch.compile(objective, backend="neuron", fullgraph=True, dynamic=False)
    # Inputs already reside on this rank's only device. Avoid DDP side streams,
    # whose global indices conflict with native per-process core visibility.
    ddp = DistributedDataParallel(compiled, device_ids=None, broadcast_buffers=False, bucket_cap_mb=64)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False)
    rows, cpu_check = [], None
    warm_started = None
    for step in range(args.steps):
        dist.barrier()
        torch.neuron.synchronize()
        if step == args.warmup:
            warm_started = time.monotonic()
        before = time.monotonic()
        b = batch_at(step, rank)
        x, y, m = (v.to(device) for v in (b.inputs, b.targets, b.target_mask))
        opt.zero_grad(set_to_none=True)
        loss = ddp(x, y, m)
        loss.backward()
        if step == 0 and cpu_reference is not None:
            batches = [batch_at(0, r) for r in range(world)]
            ref = Objective(cpu_reference, batches[0].prediction_positions)
            ref_loss = ref(torch.cat([b.inputs for b in batches]), torch.cat([b.targets for b in batches]),
                           torch.cat([b.target_mask for b in batches]))
            ref_loss.backward()
            err, total = 0., 0.
            for a, z in zip(model.parameters(), cpu_reference.parameters(), strict=True):
                err += (a.grad.detach().cpu() - z.grad).square().sum().item()
                total += z.grad.square().sum().item()
            relative_l2 = (err/max(total, 1e-30))**.5
            if relative_l2 > .02:
                raise AssertionError(f"DDP reduction differs from CPU global batch: {relative_l2}")
            cpu_check = {"cpu_global_loss": ref_loss.item(), "gradient_relative_l2": relative_l2,
                         "global_real_targets": world*args.batch*args.context}
        opt.step()
        torch.neuron.synchronize()
        dist.barrier()
        elapsed = time.monotonic() - before
        value = loss.detach().cpu().item()
        if not torch.isfinite(torch.tensor(value)):
            raise AssertionError("Nonfinite distributed loss.")
        rows.append({"step": step, "seconds": elapsed, "local_loss": value})
        if rank == 0:
            print(json.dumps(rows[-1]), flush=True)
    warm_wall = time.monotonic() - warm_started
    inference = None
    if args.inference:
        model.eval()
        with torch.no_grad():
            dist.barrier()
            torch.neuron.synchronize()
            cold_start = time.monotonic()
            for _ in range(2):
                compiled(x, y, m)
            torch.neuron.synchronize()
            dist.barrier()
            cold_seconds = time.monotonic() - cold_start
            inference_start = time.monotonic()
            for _ in range(8):
                compiled(x, y, m)
                torch.neuron.synchronize()
                dist.barrier()
            inference = {"two_cold_forwards_seconds": cold_seconds,
                         "eight_warm_forwards_seconds": time.monotonic()-inference_start}
    state_digest = digest({"model": model.state_dict(), "optimizer": opt.state_dict()})
    import torch_neuronx
    local_report = {"rank": rank, "physical_core": int(_cores[rank]), "steps": rows,
                    "model_optimizer_digest": state_digest, "warm_wall_seconds": warm_wall,
                    "fallback_ops": str(torch_neuronx.get_fallback_ops()), "inference": inference}
    (args.output/f"rank-{rank}.json").write_text(json.dumps(local_report, indent=2)+"\n")
    dist.barrier()
    if rank == 0:
        reports = [json.loads((args.output/f"rank-{r}.json").read_text()) for r in range(world)]
        if len({r["model_optimizer_digest"] for r in reports}) != 1:
            raise AssertionError("Full model and optimizer states differ across replicas.")
        wall = max(r["warm_wall_seconds"] for r in reports)
        targets = (args.steps-args.warmup)*world*args.batch*args.context
        report = {"status": "PASS_NATIVE_DDP_BENCHMARK", "scientific_result": False,
                  "world_size": world, "physical_chips": sorted({int(c)//4 for c in _cores}),
                  "parameters": config.parameter_count(), "size": args.size, "context": args.context,
                  "pauses": args.pauses, "local_batch": args.batch,
                  "warm_global_real_targets_per_second": targets/wall,
                  "warm_seconds": wall, "warm_real_targets": targets,
                  "cold_first_step_seconds": max(r["steps"][0]["seconds"] for r in reports),
                  "all_replica_model_optimizer_states_equal": True, "cpu_global_check": cpu_check,
                  "wall_seconds_rank0": time.monotonic()-start,
                  "fallback_ops_by_rank": {str(r["rank"]): r["fallback_ops"] for r in reports},
                  "flops_per_global_step": config.flops(args.context, first.inputs.shape[1], args.batch*world),
                  "scope": "FP32 parameters and arithmetic, native AdamW, synchronized DDP, natural full windows and H2D; no evaluation or checkpoint in warm timer."}
        if args.inference:
            inf_wall = max(r["inference"]["eight_warm_forwards_seconds"] for r in reports)
            report["inference"] = {"teacher_forced_real_targets_per_second": 8*world*args.batch*args.context/inf_wall,
                                   "mean_forward_seconds": inf_wall/8,
                                   "cold_two_forwards_seconds": max(r["inference"]["two_cold_forwards_seconds"] for r in reports),
                                   "scope": "Teacher-forced window NLL, preloaded inputs, synchronized replicas; no KV-cached autoregressive decoding."}
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
