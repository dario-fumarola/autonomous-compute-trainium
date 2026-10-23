"""Fresh-process, read-only DEV evaluation; TEST is deliberately inaccessible."""

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from compute_trainium.data import document_windows, layout, pad_layout
from compute_trainium.metrics import DocumentScore, summarize
from compute_trainium.model import ModelConfig
from compute_trainium.titan import build_titan
from compute_trainium.recurrence import config_from_signature, build_from_config
from compute_trainium.training import TokenObjective, file_sha256
from scripts.qualify_native import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-documents-per-domain", type=int, default=0)
    parser.add_argument("--cohort", choices=["all", "tune", "curve"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    expected = saved["signature"].get("dev_corpus_sha256", saved["signature"]["corpus_sha256"])
    if file_sha256(args.corpus) != expected:
        raise ValueError("Evaluation corpus differs from checkpoint")
    config = config_from_signature(saved["signature"])
    context, precision = saved["signature"]["context"], saved["signature"]["precision"]
    pauses, exposure = saved["pauses"], saved["real_tokens"]
    padding = saved["signature"].get("pad_to", 0)
    model = build_from_config(config)
    model.load_state_dict(saved["model"])
    before = digest(model.state_dict())
    del saved
    reference = copy.deepcopy(model).eval()
    model.to("neuron").eval()
    corpus = json.loads(args.corpus.read_text())
    docs = []
    for domain in ("web", "math", "code"):
        pool = sorted((d for d in corpus if d["split"] == "dev" and d["domain"] == domain
                       and (args.cohort == "all" or d.get("cohort") == args.cohort)), key=lambda d: d["id"])
        docs.extend(pool[:args.max_documents_per_domain] if args.max_documents_per_domain else pool)
    if not docs:
        raise ValueError("DEV is empty")
    source, mask = next(document_windows(docs[0]["tokens"], context))
    first = pad_layout(layout(source[None], mask[None], pauses=pauses), padding)
    objective = TokenObjective(model, first.prediction_positions, precision == "bf16")
    compiled = torch.compile(objective, backend="neuron", fullgraph=True, dynamic=False)
    scores, cpu_check = [], None
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for doc in docs:
            total = 0.
            count = 0
            for source, mask in document_windows(doc["tokens"], context):
                b = pad_layout(layout(source[None], mask[None], pauses=pauses), padding)
                value = compiled(*(v.contiguous().to("neuron") for v in (b.inputs, b.targets, b.target_mask)))
                loss_sum = value.detach().cpu().item()
                targets = int(mask.sum())
                if cpu_check is None:
                    torch.set_num_threads(8)
                    ref = TokenObjective(reference, b.prediction_positions, precision == "bf16", "cpu")
                    cpu_sum = ref(b.inputs, b.targets, b.target_mask).item()
                    if abs(loss_sum-cpu_sum)/targets > (.03 if precision == "bf16" else .003):
                        raise AssertionError("Fresh evaluation graph differs from CPU")
                    cpu_check = {"native_nll": loss_sum/targets, "cpu_nll": cpu_sum/targets}
                    del reference
                    torch.set_num_threads(2)
                if not torch.isfinite(torch.tensor(loss_sum)):
                    raise AssertionError("Nonfinite DEV loss")
                total += loss_sum
                count += targets
            if count != len(doc["tokens"]):
                raise AssertionError("DEV real-target coverage differs")
            scores.append(DocumentScore(doc["domain"], doc["id"], doc["group"], total, count))
    if digest(model.state_dict()) != before:
        raise AssertionError("Evaluation changed full model state")
    report = {"status": "PASS_CHECKPOINT_DEV_EVALUATION", "split": "dev", "cohort": args.cohort,
              "real_training_tokens": exposure, "pauses": pauses, "precision": precision,
              "documents": len(scores), "targets": sum(s.tokens for s in scores),
              "metrics": summarize(scores), "cpu_probe": cpu_check,
              "weights_unchanged": True, "wall_seconds": time.monotonic()-started,
              "scope": "Isolated documents; qualification subset when max-documents is nonzero",
              "max_documents_per_domain": args.max_documents_per_domain}
    # Preserve pairing/group sufficient statistics locally for later bootstrap.
    # The public collector publishes only the aggregate report.
    (args.output/"document-scores.json").write_text(json.dumps([asdict(s) for s in scores])+"\n")
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
