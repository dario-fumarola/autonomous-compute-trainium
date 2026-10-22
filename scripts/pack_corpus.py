"""Materialize deterministic packing once; workers use shared mapped arrays.

This JSON adapter is for the small qualification corpus. A large-corpus freeze
must produce this same array/manifest contract with streaming preprocessing.
"""

import argparse
import json
from pathlib import Path

from compute_trainium.training import PackedCorpus, file_sha256


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--seed", type=int, default=23)
    args = p.parse_args()
    packed = PackedCorpus(json.loads(args.corpus.read_text()), args.context, args.seed)
    packed.save_mapped(args.output, file_sha256(args.corpus))
    print(json.dumps({"windows": packed.windows, "real_targets": packed.real_targets,
                      "fingerprint": packed.fingerprint}))
