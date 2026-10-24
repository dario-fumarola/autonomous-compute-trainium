# Finite scientific-driver integration check

This is a software integration check, not a scientific comparison. Use only
qualification seed1009 initial states and mark every saved state/receipt
qualification-only. The driver must reject these states for scientific fits.
Ordinary and sampled-depth model operators already have complete FP16/FP32
global-gradient, first-update,120-update, recovery and warm100 evidence.

On released eight-chip halves, exercise the new frozen-plan driver for each
model:20 updates from qualification initialization; a fresh-process continuation
to40; and a separate shorter cooldown branch from the stable20-update state to30.
The second and third phases must restore the exact original model/optimizer/RNG
state and match source/work counters. Every checkpoint requires identical
complete replicas. The branch and main endpoint each finish their own zero-LR
cooldown. Count20+20+10 actual calls per replica/model, zero research updates.

The shared absolute warmup is the first4 updates' real exposure; main cooldown
starts at20 and ends40, short cooldown starts at20 and ends30. Peak6e-4 is
conservative and is not a selected scientific LR. Use FP16 body, FP32 head/
masters, manual8192 scale, eager FP32-polar Muon and fused unscaled AdamW.
Exact original natural targets, packing, source/depth/noise ordering and context
1024/global512 are retained. No held-out data or new random corpus is used.

Wait for the preceding profile's final native case on each half to close before
claiming it; do not steal the gap between its warm and replay phases. CPU audits
may finish concurrently.20-minute whole integration window,10minutes per native
phase,128GiB per owner/256GiB aggregate and unchanged512GiB/35GiB reserves.
Failure cancels only this check's owners unless a true shared reserve emergency
requires global stop. No scientific launch follows automatically.
