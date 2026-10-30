# Mature clipping stress check fails; candidate remains disabled

The first ordinary-model case at the saved gradient scale produces identical
gradients, moments and parameter updates. The ten-times-gradient clipping stress
case fails the registered1e-5 relative-error limit: gradient error0.000187799,
parameter-update error0.000114754, first-moment error0.000168916 and second-moment
error0.000240705. The maximum parameter difference is1.19e-7, but the frozen
relative gate is not relaxed.

The original per-parameter norm is2.641026; the concatenated-buffer norm is
2.640530. This identifies a floating-point accumulation difference in the
clipping coefficient. The small CPU fixtures did not expose the large-vector
case. The remaining six mature cases are not run after the failure.

No scientific checkpoint or optimizer state is modified. Four disposable CPU
optimizer steps are charged,9.598 wall seconds,33.791 CPU seconds and4.365 GB
peak RSS. No accelerator work or model scoring occurs. The original all16-chip
scientific control is unchanged. A repair must preserve the original norm
decomposition and rerun the same strict checks under a new recorded attempt.

[Retained failure and both completed cases](report.json) ·
[Original frozen CPU protocol](../../../docs/BUCKETED_MATURE_CPU_20261030.md).
