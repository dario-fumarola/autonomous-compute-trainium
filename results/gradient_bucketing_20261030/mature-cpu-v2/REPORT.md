# Mature CPU gradient batching repair

All eight original cases pass after preserving the original per-parameter norm
decomposition: ordinary and pause, latent K2 and K3, each at original gradient
scale and at ten times that scale to exercise clipping. Gradients, norms,
parameter updates and both AdamW moments match exactly in these CPU checks.
The original 1e-5 relative-error threshold is unchanged.

The first attempt failed because reducing the norm over a large concatenated
FP32 buffer changed its accumulation error. Its failure, source and four
disposable optimizer steps remain in [mature-cpu-v1](../mature-cpu-v1/REPORT.md).
The repaired helper batches communication, averaging and final coefficient
application while retaining the original norm calculation. Thirteen focused
tests pass, including a large-vector regression that detects the original bug.

This attempt uses 26.964 wall seconds, 121.357 CPU seconds and 4.374 GB peak RSS,
with sixteen disposable CPU optimizer steps and zero scientific updates or
assigned chips. Captured gradients were already globally averaged; this is not
an accelerator collective or speed qualification. No active trainer imports the
helper. Native parity, restart and measured whole-update cost remain outstanding.

[Frozen repair](../../../docs/BUCKETED_NORM_REPAIR_20261030.md) ·
[Complete numerical results](report.json) · [Plan](plan.json).
