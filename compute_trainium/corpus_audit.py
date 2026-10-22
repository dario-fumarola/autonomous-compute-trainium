"""Deterministic near-duplicate screening, separate from source tokenization."""

import re
import unicodedata

import numpy as np
import xxhash


def text_sketch(text):
    words = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).casefold().split()
    if len(words) < 5:
        hashes = np.array([xxhash.xxh3_64_intdigest(" ".join(words).encode())], dtype=np.uint64)
    else:
        hashes = np.unique(np.fromiter((xxhash.xxh3_64_intdigest(" ".join(words[i:i+5]).encode())
                                       for i in range(len(words)-4)), dtype=np.uint64))
    rng = np.random.default_rng(20261022)
    a = rng.integers(0, 2**64, 32, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**64, 32, dtype=np.uint64)
    with np.errstate(over="ignore"):
        signature = (a[:, None]*hashes[None, :]+b[:, None]).min(1)
    return signature, hashes[:256]


def bottom_k_jaccard(a, b, k=256):
    union = np.union1d(a, b)[:k]
    return float((np.isin(union, a, assume_unique=True) &
                  np.isin(union, b, assume_unique=True)).sum()/max(1, len(union)))


class NearDuplicateIndex:
    """32 affine minhash values, 8 bands x 4; verify with bottom-256 sketch.

    This is a disclosed approximate audit, not an exhaustive semantic duplicate
    guarantee. Reject crowded candidate buckets rather than skipping checks.
    """
    def __init__(self, threshold=.8, maximum_candidates=256):
        self.threshold, self.maximum_candidates = threshold, maximum_candidates
        self.bands, self.sketches = {}, []

    def keys(self, signature):
        return [(i, signature[4*i:4*i+4].tobytes()) for i in range(8)]

    def check(self, signature, sketch):
        candidates = set()
        for key in self.keys(signature):
            candidates.update(self.bands.get(key, ()))
            if len(candidates) > self.maximum_candidates:
                return "crowded_near_duplicate_bucket"
        for index in sorted(candidates):
            if bottom_k_jaccard(sketch, self.sketches[index]) >= self.threshold:
                return "near_duplicate"
        return None

    def add(self, signature, sketch):
        index = len(self.sketches)
        self.sketches.append(sketch)
        for key in self.keys(signature):
            self.bands.setdefault(key, []).append(index)
