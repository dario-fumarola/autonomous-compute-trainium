"""Document-level real-token scoring; DEV and TEST remain explicit choices."""

import torch

from .data import document_windows, layout
from .metrics import DocumentScore, real_token_losses


@torch.no_grad()
def evaluate(model, documents, *, split="dev", context=512, pauses=0, device="cpu"):
    model.eval()
    scores = []
    for d in documents:
        if d["split"] != split:
            continue
        total, count = 0., 0
        for source, mask in document_windows(d["tokens"], context):
            batch = layout(source[None], mask[None], pauses=pauses)
            logits = model(batch.inputs.to(device), batch.prediction_positions.to(device))
            loss = real_token_losses(logits, batch.targets.to(device), batch.target_mask.to(device))
            total += loss.double().sum().cpu().item()
            count += mask.sum().item()
        if count != len(d["tokens"]):
            raise AssertionError("Evaluation lost source targets.")
        scores.append(DocumentScore(d["domain"], d["id"], d["group"], total, count))
    return scores
