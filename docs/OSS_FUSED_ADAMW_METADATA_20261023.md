# Fused AdamW optional-metadata contract — reproducible evidence

The pinned native backend accepts `grad_scale` and `found_inf` on the direct
ATen fused-AdamW operator, but our numerical fixture finds that these arguments
do not affect its update. The same-version CPU operator honors them. Normal
unscaled updates agree. This is a narrowly scoped operator-contract result;
high-level GradScaler reachability and compiled-optimizer behavior were not
tested.

Environment: PyTorch 2.14.0+cpu, torch-neuronx 2.14.3.0.282+dc70ffa,
Trainium2 LNC=2, the isolated Ubuntu 26.04 native environment, FP32 parameters
and moments. No corpus or scientific checkpoint is used by this fixture.

| Case | Same-version CPU | Native observation |
| --- | --- | --- |
| No optional metadata | Ordinary update | Matches; parameter error 0, first-moment error 1.49e-8 |
| `found_inf=1` | Parameters and moments unchanged | Updates anyway; maximum parameter change 9.07e-5 |
| `grad_scale=4`, `found_inf=0` | First moment approximately 0.18025 | First moment approximately 0.181; maximum error 7.50e-4 |

Run in a correctly configured native environment with a free device:

```bash
NEURON_RT_VISIBLE_CORES=0 python -m scripts.repro_fused_amp_metadata --output /tmp/adam-contract
```

The script first self-checks the CPU overflow/unscale behavior. It then compares
three identical numerical cases with Neuron and exits nonzero on a mismatch.
The executed result is `REPRODUCED_FUSED_OPTIONAL_METADATA_MISMATCH`.
[Reproducer](../scripts/repro_fused_amp_metadata.py) ·
[Measured cases](../results/fast_path_20261023/amp-metadata-contract.json).

The upstream optimizer source itself describes both arguments as ignored
metadata. We checked the public main head on October 23: it remains
`50b14171affaa365f1df6aa015232427336f6a4d`, and that optimizer file matches the
local pinned source. This evidence adds an executable CPU/native parity check
for an acknowledged limitation; it does not establish that the limitation was
previously unknown.
[Pinned source](https://github.com/aws-neuron/torch-neuronx/blob/50b14171affaa365f1df6aa015232427336f6a4d/torch_neuronx/python_ops/torch_mlir/ops/optimizer.py).

A small contribution could add contract tests and explicitly reject unsupported
optional arguments until their semantics are implemented. Full support needs
tests of overflow skipping, unscaling, state/gradient mutation and the compiled
path. No installed package was patched. The completed scientific experiments
use plain AdamW and unscaled BF16, and the separately qualified fused candidate
also uses unscaled BF16 with FP32 masters and moments. These results do not
implicate this optional-argument gap in the scientific NLL comparisons.

This is evidence for owner review, prepared with Codex. The owner should check
related issues and review the reproducer before any submission. The upstream
[AI policy](https://github.com/aws-neuron/torch-neuronx/blob/main/AI_POLICY.md)
requires disclosure and human accountability and excludes fully autonomous
pull requests. No upstream issue, comment or PR has been submitted.
