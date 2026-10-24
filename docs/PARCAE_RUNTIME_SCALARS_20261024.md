# Changing-schedule scalar overhead

The first scientific screen is stopped before any held-out scoring: measured
changing-scalar updates take about4.4seconds, whereas constant-scalar updates
take1.1–1.2seconds. Projected warmup/cooldown work does not fit the frozen25minute
phase and3hour screen caps. Keep its original plan,20-update checkpoints,
partial-update logs and all costs. Do not extend that execution or change its
code silently. No rate or scientific quality conclusion follows from this stop.

The installed native beta registers `aten::fill_.Scalar` with value argument1
static. Muon's transient LR/momentum/decay buffers used this path. Fused AdamW's
LR is already a runtime input according to the installed registration. The
observed timing and source inspection motivate a smaller change: pack exactly
the same FP32 scalars into a CPU tensor and copy it to the device as runtime data.
Keep all mathematical operations,moment layout,grouping,initialization,data,
rates and schedules identical. This is a performance hypothesis until measured.

CPU tests require bit-identical complete parameters and optimizer state across
three changing LR/momentum/decay updates and exact checkpoint recovery. Native
qualification runs all three actual model inventories at the common FP16/global512
geometry. Each case has20 warmup updates and100 timed updates with distinct LR
and decay values,then an uninterrupted121st step and a fresh-process replay.
Preserve the existing independent first-actual-update/global-trained-gradient,
named-family2%,loss0.003,replica/fallback and complete-state recovery gates.
The first conservative20-update schedule is unchanged for the independent
first-update checker. No held-out scoring or scientific weight reuse occurs.

Bound the three-case qualification to45minutes,each native phase10minutes and
each CPU audit20minutes. Use all16 chips as two owned eight-chip jobs,explicit
non-ephemeral communication ports,128GiB caps and the unchanged host/disk reserves.
Only if this passes may a separately registered scientific screen use the new
scalar ingestion. Recompute its budget using changing-schedule measurements.
Retain all unsuccessful attempts; prepare an OSS reproducer without submitting
or changing the installed SDK.
