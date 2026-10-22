"""Single-core CPU-checked inference timing, isolated from training graph memory."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from compute_trainium.data import document_windows, layout
from compute_trainium.model import ModelConfig
from compute_trainium.titan import build_titan
from scripts.qualify_native import Objective, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", type=int, default=512)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = ModelConfig(**json.loads((Path(__file__).resolve().parents[1]/"configs/models.json").read_text())["small"])
    torch.manual_seed(23)
    model = build_titan(config).eval()
    reference = copy.deepcopy(model)
    initial = digest(model.state_dict())
    model.to("neuron")
    docs = json.loads(args.corpus.read_text())
    doc = next(d for d in docs if d["split"] == "train" and d["domain"] == "web" and len(d["tokens"]) > args.context)
    source, mask = next(document_windows(doc["tokens"], args.context))
    rows = {}
    with torch.no_grad():
        for pauses in (0, 16):
            b = layout(source[None], mask[None], pauses=pauses)
            expected = Objective(reference, b.prediction_positions)(b.inputs, b.targets, b.target_mask).item()
            obj = torch.compile(Objective(model, b.prediction_positions), backend="neuron", dynamic=False, fullgraph=True)
            x, y, m = (v.to("neuron") for v in (b.inputs, b.targets, b.target_mask))
            cold_start = time.monotonic()
            for _ in range(2):
                actual = obj(x, y, m)
                torch.neuron.synchronize()
            cold_seconds = time.monotonic()-cold_start
            torch.testing.assert_close(actual.cpu(), torch.tensor(expected), rtol=2e-4, atol=2e-3)
            before = time.monotonic()
            for _ in range(8):
                obj(x, y, m)
                torch.neuron.synchronize()
            seconds = time.monotonic()-before
            rows[str(pauses)] = {"mean_forward_seconds": seconds/8,
                "teacher_forced_real_targets_per_second": 8*args.context/seconds,
                "cold_two_forwards_seconds": cold_seconds,
                "cpu_nll": expected, "native_nll": actual.cpu().item(),
                "expanded_positions": b.inputs.shape[1]}
            print(json.dumps({"pauses": pauses, **rows[str(pauses)]}), flush=True)
    if digest(model.state_dict()) != initial:
        raise AssertionError("Inference changed model state.")
    import torch_neuronx
    report = {"status": "PASS_NATIVE_INFERENCE", "scientific_result": False,
              "parameters": config.parameter_count(), "context": args.context,
              "rows": rows, "weights_unchanged": True,
              "fallback_ops": str(torch_neuronx.get_fallback_ops()),
              "scope": "Single-core teacher-forced real-token NLL with preloaded inputs; no KV-cached autoregressive decoding."}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
