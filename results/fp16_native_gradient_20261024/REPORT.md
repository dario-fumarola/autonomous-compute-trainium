# Native FP16 full-gradient qualification

Complete, with zero optimizer updates and no held-out scoring. Both mature
256-window objectives pass the registered2% gradient /0.003-nat NLL gates.

| Model | Gradient relative L2 error | Native minus CPU FP32 NLL |
| --- | ---: | ---: |
| Ordinary, actual update9216 | 0.004227803 | +0.000008458 |
| Fixed, update7680 | 0.011733262 | +0.000020489 |

FP16 body, FP32 vocabulary projection/loss, FP32 master gradients and manual
loss scale8192, unscaled after distributed averaging. All replica gradients
agree; source checkpoint hashes and weights remain unchanged. The ordinary
source directory contains10368 in its name; the actual stored update is9216.
No unexpected fallback. Paired captures use all16 chips in separately owned
containers and complete in181.889s, including reference comparison.

This qualifies a gradient candidate, not optimizer updates, overflow handling
during learning, recovery, steady throughput or model quality. Active scientific
training remains FP32. No TEST access, terminal attempt or scientific success.

[Exact comparisons](report.json) · [Execution receipts](execution-costs.json) ·
[Frozen protocol](../../docs/FP16_NATIVE_GRADIENT_PROTOCOL_20261024.md).
