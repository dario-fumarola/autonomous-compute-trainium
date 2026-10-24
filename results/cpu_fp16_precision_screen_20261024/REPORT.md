# CPU precision screen: a prospective FP16 candidate

On eight pinned actual TRAIN windows from the unchanged fixed2B checkpoint,
keeping the vocabulary head in FP32 and using FP16 inside the model gives
8.8556% relative gradient error without scaling. Fixed loss scaling8192 reduces
that error to **0.7251%**, below the prospectively registered2% CPU-subset screen.
Every gradient remains finite. The forward NLL differs from FP32 by0.000019960.

The independent CPU model's parameters and the source checkpoint remain unchanged.
No optimizer was instantiated, no model update ran, no held-out panel was scored
and zero native chips were assigned. Total CPU-screen wall time was
105.36 seconds, overlapping the registered native training.

This is a reason to qualify native FP16 later, not evidence of supported native
operators, a full256-window gradient, reliable optimizer scaling/recovery,
throughput or a learning benefit. All active scientific jobs retain their frozen
FP32 execution. Any future native test and adoption need a separate bounded
identity and measured budget. The failed unscaled case is preserved.

[Registration](registration.json) · [Complete measurements](report.json) ·
[Source](../../scripts/screen_cpu_fp16_precision.py).
