# Low-depth training engine: prepared, native execution suspended

CPU preparation completes in13.669s from launch,12.190s after imports, with zero
chips, optimizer updates or held-out scoring. Every shared backbone tensor is
cloned across the140,781,888-parameter ordinary and141,963,840-parameter family.
The deterministic one-pass and sampled family have identical complete initial
weights. Per-sequence sampling follows the pinned release at lambda2, minimum1,
last1 differentiated step. Expected executed mean is2.135335, not exactly2;
saved accumulation8/16 tables have means2.140593/2.135137 and maxima9/10.

The registered hardware engine covers ordinary, true one-pass and sampled
low-depth training at context1024; accumulation16 for every arm and8 additionally
for the family. Each eligible precision receives20 warmup plus100 measured warm
updates, a121st update and fresh-process replay. It checks the actual first
optimizer step, independent upstream full-objective gradients after120 updates,
complete state/RNG recovery, replica equality and measured target throughput.
Qualification weights can never initialize a scientific fit.

**Before any native phase, this controller was suspended.** The separate early
optimizer screen fails the unchanged2% Muon-family update gate at about10% error.
All39 frozen engine hashes are unchanged; no training qualification update ran.
A corrected execution and separately frozen numerical qualification are needed.
The original plan, suspension and costs remain available. This avoids repeating
a known failure while the existing scientific fits continue unchanged.

The earlier eight-pass CPU preparation is preserved separately. It never had
a native execution plan and never ran on hardware. The scientific direction is
now low depth with two recurrent blocks, not an eight-pass main experiment.

[CPU preparation](cpu-report.json) · [Original native plan](native-plan.json) ·
[Suspension](early-screen-suspension.json) ·
[Hardware protocol](../../docs/PARCAE_TRAINING_QUALIFICATION_20261024.md) ·
[Scientific charter](../../docs/LOW_DEPTH_SCIENTIFIC_PILOT_20261024.md).
