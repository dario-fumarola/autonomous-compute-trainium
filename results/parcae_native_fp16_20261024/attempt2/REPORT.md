# FP16 body gradient queue, revision2

This revision stopped at its failed recovery prerequisite before any native
phase. [Revision3](../attempt3/REPORT.md) preserves model/thresholds and changes
only that prerequisite and host caps.

The same zero-update model/gradient qualification is re-registered behind the
explicit FP32-polar optimizer/recovery gate. The old BF16-optimizer-dependent
revision is suspended and preserved. This queue reserves no chips while waiting.
Both ordinary/recurrent FP32 native global gradients have already passed.

No precision is adopted by this registration. Local/global/family2% gradients,
0.003 NLL difference, manual loss scale8192, unchanged FP32 head/masters,
replica equality and no unexpected fallback remain mandatory. The model CPU
reference and objective are unchanged. All32 source hashes are frozen. Latest
phase start03:00UTC October25, total30-minute native cap.

[Plan](plan.json) · [Live status](status.json) ·
[Precision amendment](../../../docs/PARCAE_PRECISE_EXECUTION_20261024.md) ·
[Preserved unstarted registration](unstarted-registration/failure.json).
