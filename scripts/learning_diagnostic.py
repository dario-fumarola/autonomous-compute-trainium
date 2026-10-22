"""Bounded three-arm learning diagnostic on natural documents and fresh DEV."""

import argparse
import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from compute_trainium.data import document_windows, layout
from compute_trainium.evaluate import evaluate
from compute_trainium.metrics import real_token_losses, summarize
from compute_trainium.model import ModelConfig
from compute_trainium.titan import build_titan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    documents = json.loads(args.corpus.read_text())
    config = ModelConfig(dim=128, layers=2, heads=4, kv_heads=2,
                         ffn_dim=256, vocab_size=32000, max_seq=1024)
    context, batch_size = 256, 4
    torch.manual_seed(23)
    base = build_titan(config)
    pools = {}
    for domain in ("web", "math", "code"):
        pools[domain] = [(s, m) for d in documents if d["domain"] == domain and d["split"] == "train"
                         for s, m in document_windows(d["tokens"], context)]
        random.Random("23:" + domain).shuffle(pools[domain])
    domain_order = ["web"] * 6 + ["math"] * 2 + ["code"] * 2
    cursors = {d: 0 for d in pools}
    batches = []
    for i in range(args.steps):
        examples = []
        for j in range(batch_size):
            domain = domain_order[(i * batch_size + j) % 10]
            examples.append(pools[domain][cursors[domain] % len(pools[domain])])
            cursors[domain] += 1
        batches.append((torch.stack([x[0] for x in examples]), torch.stack([x[1] for x in examples])))
    dev = []
    for domain in ("web", "math", "code"):
        dev.extend([d for d in documents if d["split"] == "dev" and d["domain"] == domain][:4])
    assert {d["group"] for d in documents if d["split"] == "train"}.isdisjoint({d["group"] for d in dev})
    report = {"purpose": "small CPU learning diagnostic, not a scientific comparison",
              "config": asdict(config), "seed": 23, "steps_per_arm": args.steps,
              "real_targets_per_arm": sum(m.sum().item() for _, m in batches),
              "test_evaluated": False, "arms": {}}
    for arm in ("ordinary", "early", "late"):
        model = copy.deepcopy(base)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
        initial_pauses = 16 if arm == "early" else 0
        before = summarize(evaluate(model, dev, context=context, pauses=initial_pauses))
        rows = []
        for i, (source, mask) in enumerate(batches):
            count = 16 if arm == "early" or arm == "late" and i >= args.steps // 2 else 0
            b = layout(source, mask, pauses=count)
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = real_token_losses(model(b.inputs, b.prediction_positions), b.targets, mask).sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            rows.append({"step": i, "loss": loss.item(), "pauses": count})
        after = summarize(evaluate(model, dev, context=context, pauses=16 if arm != "ordinary" else 0))
        report["arms"][arm] = {"dev_before": before, "dev_after": after, "training": rows}
        print(json.dumps({"arm": arm, "dev_before": before["weighted_nll"],
                          "dev_after": after["weighted_nll"]}), flush=True)
    report["wall_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
