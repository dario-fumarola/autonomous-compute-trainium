"""Deterministic packing and token-based training state, shared with qualification."""

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import layout, prediction_spans
from .metrics import real_token_losses


class PackedCorpus:
    """Causal document packing with BOS/EOS, including every source target once.

    Attention may cross document boundaries within a window. No boundary token is
    a target. DEV uses isolated documents; this deliberate context difference is
    shared across arms. Windows overlap one INPUT token, never a target.
    """
    def __init__(self, documents, context, seed=23):
        if context < 1:
            raise ValueError("Positive context required")
        self.context, self.seed = context, seed
        pools = {}
        for domain in ("web", "math", "code"):
            pools[domain] = sorted((d for d in documents if d["split"] == "train"
                                    and d["domain"] == domain), key=lambda d: d["id"])
            random.Random(f"{seed}:{domain}").shuffle(pools[domain])
        weights = {"web": .6, "math": .2, "code": .2}
        served, cursors = dict.fromkeys(pools, 0), dict.fromkeys(pools, 0)
        tokens, domains, order = [], [], []
        while any(cursors[d] < len(pools[d]) for d in pools):
            domain = min((d for d in pools if cursors[d] < len(pools[d])),
                         key=lambda d: (served[d]/weights[d], d))
            doc = pools[domain][cursors[domain]]
            cursors[domain] += 1
            payload = doc["tokens"]
            if not payload:
                raise ValueError("Empty source document")
            tokens.extend([1, *payload, 2])
            domains.extend([0, *([1 + list(pools).index(domain)]*len(payload)), 0])
            served[domain] += len(payload)
            order.append(doc["id"])
        if not tokens:
            raise ValueError("No training documents")
        # A last overlap token is needed even when stream length is context*k.
        self.windows = math.ceil((len(tokens)-1)/context)
        missing = self.windows*context + 1 - len(tokens)
        self.tokens = torch.tensor(tokens + [2]*missing, dtype=torch.int32)
        self.domains = torch.tensor(domains + [0]*missing, dtype=torch.uint8)
        self.domain_counts = served
        self.real_targets = sum(served.values())
        self.fingerprint = hashlib.sha256(self.tokens.numpy().tobytes()
                                          + self.domains.numpy().tobytes()).hexdigest()
        self.order_fingerprint = hashlib.sha256("\n".join(order).encode()).hexdigest()
        labels = self.domains[1:].view(self.windows, context)
        counts = torch.stack([(labels == d).sum(1) for d in (1, 2, 3)], dim=1)
        self.prefix_counts = torch.cat([torch.zeros(1, 3, dtype=torch.int64), counts.cumsum(0)])

    def save_mapped(self, directory, source_sha256):
        """Materialize once, outside Git; 64 workers share file-backed pages."""
        import json
        import numpy as np
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("tokens", "domains", "prefix_counts"):
            np.save(directory/(name+".npy"), getattr(self, name).numpy())
        metadata = {k: getattr(self, k) for k in ("context", "seed", "windows", "real_targets",
                    "domain_counts", "fingerprint", "order_fingerprint")}
        metadata.update(source_sha256=source_sha256,
                        files={p.name: file_sha256(p) for p in directory.glob("*.npy")})
        (directory/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")

    @classmethod
    def from_mapped(cls, directory, context, seed, source_sha256):
        import json
        import numpy as np
        directory = Path(directory)
        metadata = json.loads((directory/"manifest.json").read_text())
        if (metadata["context"], metadata["seed"], metadata["source_sha256"]) != (context, seed, source_sha256):
            raise ValueError("Mapped corpus provenance/context/seed mismatch")
        for name, expected in metadata["files"].items():
            if Path(name).name != name or file_sha256(directory/name) != expected:
                raise ValueError("Mapped corpus file fingerprint mismatch")
        result = cls.__new__(cls)
        for key in ("context", "seed", "windows", "real_targets", "domain_counts", "fingerprint", "order_fingerprint"):
            setattr(result, key, metadata[key])
        for name in ("tokens", "domains", "prefix_counts"):
            # Copy-on-write mappings share source pages and never modify files.
            setattr(result, name, torch.from_numpy(np.load(directory/(name+".npy"), mmap_mode="c")))
        if (result.tokens.shape != (result.windows*context+1,) or
                result.domains.shape != result.tokens.shape or
                result.prefix_counts.shape != (result.windows+1, 3)):
            raise ValueError("Invalid mapped packing dimensions")
        return result

    def counts(self, cursor, windows, repeat=False):
        if cursor < 0 or windows < 1 or (not repeat and cursor+windows > self.windows):
            raise ValueError("Unique natural corpus exhausted; prepare more data")
        def prefix(index):
            cycles, remainder = divmod(index, self.windows)
            return self.prefix_counts[-1]*cycles + self.prefix_counts[remainder]
        return (prefix(cursor+windows)-prefix(cursor)).tolist()

    def batch(self, indices, pauses=0, repeat=False):
        if not repeat and any(i < 0 or i >= self.windows for i in indices):
            raise ValueError("Unique natural corpus exhausted; prepare more data")
        sources, masks, labels = [], [], []
        for i in indices:
            if i < 0:
                raise ValueError("Negative stream cursor")
            start = (i % self.windows)*self.context
            sources.append(self.tokens[start:start+self.context+1].long())
            domain = self.domains[start+1:start+self.context+1]
            masks.append(domain > 0)
            labels.append(domain)
        batch = layout(torch.stack(sources), torch.stack(masks), pauses=pauses)
        return batch, torch.stack(labels)


def global_indices(cursor, micro, rank, world, batch):
    """Global stream order is independent of arm, checkpoint and process restart."""
    start = cursor + (micro*world + rank)*batch
    return list(range(start, start+batch))


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


class TokenObjective(torch.nn.Module):
    def __init__(self, model, positions, bf16=False, device="neuron"):
        super().__init__()
        self.model, self.spans = model, prediction_spans(positions)
        self.bf16, self.device_type = bf16, device

    def forward(self, inputs, targets, mask):
        with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.bf16):
            logits = self.model(inputs, prediction_spans=self.spans)
        # FP32 full vocabulary NLL, even with BF16 matmuls.
        return real_token_losses(logits, targets, mask).sum()


@dataclass(frozen=True)
class WSDSchedule:
    peak: float
    warmup_tokens: int
    decay_start: int | None = None
    decay_end: int | None = None
    floor_ratio: float = .1

    def __post_init__(self):
        if self.peak <= 0 or self.warmup_tokens < 0 or not 0 <= self.floor_ratio <= 1:
            raise ValueError("Invalid WSD schedule")
        if (self.decay_start is None) != (self.decay_end is None):
            raise ValueError("Both decay boundaries required")
        if self.decay_start is not None and not self.warmup_tokens <= self.decay_start < self.decay_end:
            raise ValueError("Decay must follow warmup")

    def lr(self, tokens):
        if tokens < self.warmup_tokens:
            return self.peak*max(1, tokens)/max(1, self.warmup_tokens)
        if self.decay_start is None or tokens <= self.decay_start:
            return self.peak
        fraction = min(1., (tokens-self.decay_start)/(self.decay_end-self.decay_start))
        return self.peak*(self.floor_ratio + (1-self.floor_ratio)*.5*(1+math.cos(math.pi*fraction)))


def cpu_tree(value):
    from torch.utils._pytree import tree_map
    return tree_map(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, value)


def atomic_checkpoint(path, state):
    """Replace one full state only after a completed, validated optimizer step."""
    import os
    from pathlib import Path
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(cpu_tree(state), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
