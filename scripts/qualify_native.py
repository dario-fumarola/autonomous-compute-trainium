"""Bounded hardware checks on natural text, distinct from scientific training."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

from compute_trainium.data import document_windows, layout, prediction_spans
from compute_trainium.metrics import real_token_losses
from compute_trainium.model import LanguageModel, ModelConfig
from compute_trainium.titan import build_titan


def digest(state):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            value = x.detach().cpu().contiguous()
            h.update(str((value.dtype, value.shape)).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return x
    # Include scalar hyperparameters and dictionary structure as well as tensors.
    h.update(str(tree_map(lambda x: (str(x.dtype), tuple(x.shape)) if isinstance(x, torch.Tensor) else x, state)).encode())
    tree_map(visit, state)
    return h.hexdigest()


class Objective(torch.nn.Module):
    def __init__(self, model, prediction_positions):
        super().__init__()
        self.model = model
        self.spans = prediction_spans(prediction_positions)

    def forward(self, inputs, targets, mask):
        logits = self.model(inputs, prediction_spans=self.spans)
        return real_token_losses(logits, targets, mask).sum() / mask.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", choices=["tiny", "small", "medium", "large"], default="tiny")
    p.add_argument("--engine", choices=["reference", "titan"], default="titan")
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--arm", choices=["ordinary", "early", "late"], default="ordinary")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    if args.steps < 6:
        raise ValueError("Qualification requires at least six updates.")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import torch_neuronx
    device = "neuron"
    config = ModelConfig(dim=128, layers=2, heads=4, kv_heads=2, ffn_dim=256,
                         vocab_size=32000, max_seq=1024) if args.size == "tiny" else ModelConfig(**json.loads(
                             (Path(__file__).resolve().parents[1] / "configs/models.json").read_text())[args.size])
    torch.manual_seed(23)
    model = build_titan(config) if args.engine == "titan" else LanguageModel(config)
    assert config.parameter_count() == sum(p.numel() for p in model.parameters())
    cpu_model = copy.deepcopy(model)
    corpus = json.loads(args.corpus.read_text())
    selected = []
    for domain in ["web", "math", "code"]:
        for d in corpus:
            if d["split"] == "train" and d["domain"] == domain and len(d["tokens"]) > args.context:
                source, mask = next(document_windows(d["tokens"], args.context))
                selected.append((source[None], mask[None], d["id"]))
                break
    if len(selected) != 3:
        raise ValueError("Qualification needs a natural training window in each domain.")
    initial = digest(model.state_dict())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False)
    objectives = {}
    for count in ([0, 16] if args.arm == "late" else [16] if args.arm == "early" else [0]):
        batch = layout(selected[0][0], selected[0][1], pauses=count)
        obj = Objective(model, batch.prediction_positions.to(device))
        objectives[count] = torch.compile(obj, backend="neuron", dynamic=False, fullgraph=True) if args.compile else obj
    rows = []
    def step(index):
        count = 16 if args.arm == "early" or (args.arm == "late" and index >= args.steps // 2) else 0
        source, mask, key = selected[index % 3]
        batch = layout(source, mask, pauses=count)
        inputs, targets, mask = (x.to(device) for x in (batch.inputs, batch.targets, batch.target_mask))
        optimizer.zero_grad(set_to_none=True)
        torch.neuron.synchronize()
        before = time.monotonic()
        loss = objectives[count](inputs, targets, mask)
        loss.backward()
        if index == 0 and not rows:
            cpu_batch = layout(source, mask.cpu(), pauses=count)
            cpu_loss = Objective(cpu_model, cpu_batch.prediction_positions)(cpu_batch.inputs, cpu_batch.targets, cpu_batch.target_mask)
            cpu_loss.backward()
            torch.testing.assert_close(loss.detach().cpu(), cpu_loss.detach(), atol=2e-3, rtol=2e-4)
            max_error, error_sq, ref_sq = 0., 0., 0.
            for (_, a), (_, b) in zip(model.named_parameters(), cpu_model.named_parameters(), strict=True):
                grad = a.grad.detach().cpu()
                delta = (grad - b.grad).float()
                max_error = max(max_error, delta.abs().max().item())
                error_sq += delta.square().sum().item()
                ref_sq += b.grad.float().square().sum().item()
            relative_l2 = (error_sq / max(ref_sq, 1e-30)) ** .5
            if relative_l2 > .02:
                raise AssertionError(f"Native gradient mismatch: {relative_l2}")
            (args.output / "cpu-agreement.json").write_text(json.dumps({"cpu_loss": cpu_loss.item(),
                "native_loss": loss.item(), "max_gradient_error": max_error,
                "relative_gradient_l2": relative_l2}, indent=2))
        optimizer.step()
        torch.neuron.synchronize()
        elapsed = time.monotonic() - before
        row = {"step": index, "pauses": count, "loss": loss.detach().cpu().item(),
               "seconds": elapsed, "real_targets": batch.target_mask.sum().item(),
               "expanded_positions": batch.inputs.shape[1], "source_document": key}
        if not torch.isfinite(loss.detach()).cpu().item():
            raise AssertionError("Nonfinite native loss.")
        rows.append(row)
        print(json.dumps(row), flush=True)
    checkpoint_at = args.steps // 2
    checkpoint = args.output / "checkpoint.pt"
    checkpoint_wall = None
    for i in range(args.steps):
        if i == checkpoint_at:
            t = time.monotonic()
            state = tree_map(lambda x: x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x,
                             {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "next_step": i})
            torch.save(state, checkpoint)
            checkpoint_wall = time.monotonic() - t
        step(i)
        if i == checkpoint_at:
            expected = digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            expected_loss = rows[-1]["loss"]
            recovered = torch.load(checkpoint, weights_only=False)
            model.load_state_dict(recovered["model"])
            optimizer.load_state_dict(recovered["optimizer"])
            step(recovered["next_step"])
            actual = digest({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            if actual != expected or rows[-1]["loss"] != expected_loss:
                raise AssertionError("Full model/optimizer recovery does not replay exactly.")
    assert initial != digest(model.state_dict())
    warm = [r for r in rows if r["step"] >= checkpoint_at + 1]
    report = {"status": "PASS_NATIVE_QUALIFICATION", "scientific_result": False,
              "config": asdict(config), "engine": args.engine, "arm": args.arm,
              "compiled": args.compile, "parameters": config.parameter_count(),
              "initial_state_sha256": initial, "full_optimizer_replay": True,
              "checkpoint_seconds": checkpoint_wall, "updates_including_replay": len(rows),
              "warm_real_targets_per_second": sum(r["real_targets"] for r in warm) / sum(r["seconds"] for r in warm),
              "wall_seconds": time.monotonic() - started, "steps": rows,
              "versions": {k: importlib.metadata.version(k) for k in ["torch", "torch-neuronx", "neuronx-cc", "nki"]},
              "fallback_ops": str(torch_neuronx.get_fallback_ops()),
              "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}), flush=True)


if __name__ == "__main__":
    main()
