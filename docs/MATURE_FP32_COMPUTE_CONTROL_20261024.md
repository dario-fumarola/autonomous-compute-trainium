# Resource control for the paired FP32 three-billion diagnostic

Identity: `mature-fp32-compute-v1`. Development only, one reused paired seed103.
No new TEST, hypothesis selection, seed interval or terminal alpha allocation.

Complete the missing resource comparison: fixed recurrence at update11520
(3,014,637,826 original targets) versus ordinary at update19490
(5,100,221,387 targets). The estimated dense training matmul totals differ by
0.00306%. Keep that approximation explicit; include vocabulary projection and
backward. This is not a wall-time or precision-weighted hardware-FLOP match.

Continue the ordinary model from the independent-audited **stable** FP32
update10368 state of `mature-fp32-v1`. Do not continue the decayed3B endpoint.
Preserve full model, optimizer moments, RNG, data cursor and counters. Same
150,436,608 parameters, corpus, source order, routing, global256 windows,
FP32 primitive attention, fused unscaled AdamW, clipping and peak LR6e-4.
The histories below2.010B remain BF16 in both models; later work is FP32.

Extend the stable schedule to update17541 and decay through19490, using the
exact immutable packed-prefix target counts. Score both checkpoints on the
existing consumed DEV panel. The final completed-decay comparison is primary
for this one-pair resource diagnostic; the stable point is descriptive.
Independent whole-panel CPU FP32 scores and serialized full-state copy audits
are required. Existing0.0002-nat score and2% local gradient gates remain.

Use half0,32 ranks, batch1, accumulation8. Wait until the prior ordinary3B
native evaluation has closed before claiming those cores. The other half may
finish its registered3B fit or run separately registered useful work. All
overlapping assignments are recorded; no exclusive-node accounting is assumed.

Budget from the measured ordinary176,468 targets/s, plus195s per training
launch and100s per readout,30% margin and a30-minute independent CPU tail.
Freeze source, parent, data and panel hashes before updates. Training cutoff
14:45 UTC and hard deadline15:30 UTC, October24. Refuse a late launch that no
longer fits. Full checkpoints every1024 updates, plus each phase endpoint.
Failures close this identity honestly; recovery requires a new explicit plan.

Use the existing numerical recovery policy: exact complete state on load,
identical final replicas and correct counters. Future process-restart
arithmetic is not promised bit-identical. Preserve every earlier stopped run.
This strong ordinary control remains necessary even if recurrence improves at
equal data. A deficit here is not erased by the equal-data result.
