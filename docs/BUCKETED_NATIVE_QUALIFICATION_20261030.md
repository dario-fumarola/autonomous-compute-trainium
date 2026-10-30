# Bounded Native gradient-batching qualification

This is disposable engineering qualification, not another scientific fit. It
may start only after the active latent-exposure matched-work controller closes,
including its audits and readouts. It cannot interrupt or alter that comparison.
No qualification weights, source repeats or scores become research endpoints.

Use the already qualified tied ordinary shape: 11 layers, width512, FFN1376,
32k vocabulary, context512, 51,179,008 parameters; 64 ranks on16 chips, local
batch2, AdamW lr0.0003, betas0.9/0.95, epsilon1e-8, decay0.1 with the existing
table/norm exemptions. Seed9841 and original natural TRAIN qualification data.
Only reduction/clipping implementation differs. Freeze the source bundle and
data hashes before launch. The repaired32MiB batching helper preserves the
original per-parameter norm decomposition; its eight mature CPU checks pass.

In one worker allocation, run original/bucketed/bucketed/original (ABBA).
Reset the same model, optimizer, RNG and data order before each trial. Each
trial has three untimed compile/startup updates then100 whole warm updates;
timing includes data transfer, forward, backward, averaging, clipping, fused
AdamW and loss synchronization. All trials use exactly the same windows.
Compilation, captures and intertrial resets are reported separately. Record
both order-specific time ratios; a speed qualification requires at least5%
lower mean whole-update time in both comparisons, not just a microbenchmark.
Numerical success without this speed gate does not promote the optimization.

Retain the initial model, complete first-update states and averaged gradients
for the first original and bucketed trials. Compare all gradient families,
updates relative to step size, and both moments. Independent CPU cross-entropy
on the complete128-window first batch must match each path within the existing
2% overall/family gradient and update/moment gates and0.003 loss tolerance.
Also retain bucketed state103, its complete pre-update104 gradient and state104.
Require the independent mature128-window CPU gradient gate and exact64-replica
full-state equality. A separate worker must restore model/optimizer/RNG at103
and reproduce104 exactly. Unexpected fallbacks or compiler errors stop the
qualification. Original-vs-bucketed Native first-step differences use the same
2% family/update/moment gates, with actual errors reported.

Limits:90 minutes total after admission,30 minutes per Native allocation,
20 minutes CPU audit,192GiB worker cap,512GiB host MemAvailable reserve,
35GiB free per output filesystem, projected inodes plus10,000 reserve, and
4GiB artifact allowance. Preserve failures and their costs. No automatic
scientific adoption, latent-graph speed claim or change to matrix-work ledgers.
