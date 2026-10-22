"""Numerical software fixture: native compiled input-prologue strided fallback."""

import argparse
import json
from pathlib import Path

import torch
import torch_neuronx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", help="Unused common launcher argument")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contiguous", action="store_true")
    args = p.parse_args()
    # Mimics a two-row token batch with one overlap column removed.
    source = torch.arange(2*1025, dtype=torch.int64).reshape(2, 1025)[:, :1024]
    if args.contiguous:
        source = source.contiguous()
    compiled = torch.compile(lambda x: (x.float()*x.float()).sum(), backend="neuron",
                             fullgraph=True, dynamic=False)
    value = compiled(source.to("neuron")).cpu()
    torch.testing.assert_close(value, (source.float()**2).sum())
    report = {"status": "PASS_STRIDED_INPUT_NUMERICS", "shape": list(source.shape),
              "cpu_stride": list(source.stride()), "host_contiguous": args.contiguous,
              "fallback_ops": str(torch_neuronx.get_fallback_ops()),
              "numerical_agreement": True,
              "scope": "Integer array software fixture, never research data"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
