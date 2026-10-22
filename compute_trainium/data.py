"""Document isolation, fixed windows, and causal pause alignment."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

import torch


def content_hash(text: str) -> str:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def source_group(domain: str, source: str) -> str:
    if domain == "code":
        repo = source.strip().lower().removesuffix(".git")
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError("A code document needs an owner/repository source.")
        return "repo:" + repo
    host = (urlsplit(source).hostname or "").lower().removeprefix("www.")
    if not host:
        raise ValueError("A web document needs a source URL.")
    return "host:" + host


def split_for_group(group: str) -> str:
    # A shared host gets one split across both web and math domains.
    bucket = int(hashlib.sha256(("natural-v1:" + group).encode()).hexdigest()[:16], 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def document_windows(tokens: list[int], context: int, bos: int = 1, eos: int = 2):
    """Score every source token once; boundary tokens and padding carry no loss."""
    if context < 1 or not tokens:
        raise ValueError("Pass a positive context and a nonempty document.")
    sequence = [bos, *tokens, eos]
    for start in range(0, len(sequence) - 1, context):
        chunk = sequence[start:start + context + 1]
        valid = [start + j <= len(tokens) for j in range(1, len(chunk))]
        if not any(valid):
            continue
        missing = context + 1 - len(chunk)
        yield (torch.tensor(chunk + [eos] * missing),
               torch.tensor(valid + [False] * missing, dtype=torch.bool))


@dataclass
class Layout:
    inputs: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    prediction_positions: torch.Tensor
    pause_mask: torch.Tensor


def pad_layout(batch: Layout, length: int = 0, pad_id: int = 2) -> Layout:
    """Right-tail padding cannot affect earlier causal predictions or targets."""
    if not length:
        return batch
    missing = length-batch.inputs.shape[1]
    if missing < 0:
        raise ValueError("Padding length is smaller than the expanded sequence")
    return Layout(torch.nn.functional.pad(batch.inputs, (0, missing), value=pad_id),
                  batch.targets, batch.target_mask, batch.prediction_positions,
                  torch.nn.functional.pad(batch.pause_mask, (0, missing), value=False))


def layout(source: torch.Tensor, target_mask: torch.Tensor, *, pauses: int = 0,
           interval: int = 128, pause_id: int = 0) -> Layout:
    """Map target j to expanded position p(j)-1, including the last pause."""
    if source.ndim != 2 or target_mask.shape != (source.shape[0], source.shape[1] - 1):
        raise ValueError("Source and target mask shapes do not match.")
    if pauses < 0 or interval < 1:
        raise ValueError("Pause count must be nonnegative and interval positive.")
    parts, marks, positions = [], [], []
    index = 0
    for j in range(source.shape[1]):
        if j and j % interval == 0:
            parts.extend([torch.full_like(source[:, :1], pause_id)] * pauses)
            marks.extend([True] * pauses)
            index += pauses
        if j:
            positions.append(index - 1)
        parts.append(source[:, j:j + 1])
        marks.append(False)
        index += 1
    expanded = torch.cat(parts, dim=1)
    return Layout(expanded[:, :-1], source[:, 1:], target_mask,
                  torch.tensor(positions, device=source.device),
                  torch.tensor(marks[:-1], device=source.device))


def dummy_pause_mask(pause_mask: torch.Tensor) -> torch.Tensor:
    n = pause_mask.numel()
    causal = torch.ones((n, n), dtype=torch.bool, device=pause_mask.device).tril()
    return causal & (~pause_mask[:, None] | pause_mask[None, :])


def prediction_spans(positions: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Represent a static ordered gather as contiguous slices for compilation."""
    values = positions.detach().cpu().tolist()
    if not values or values != sorted(set(values)):
        raise ValueError("Prediction positions must be nonempty and strictly increasing.")
    spans, start, previous = [], values[0], values[0]
    for value in values[1:]:
        if value != previous + 1:
            spans.append((start, previous+1))
            start = value
        previous = value
    spans.append((start, previous+1))
    return tuple(spans)
