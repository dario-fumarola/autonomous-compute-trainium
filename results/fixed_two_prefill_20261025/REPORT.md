# Prefill cost: registered queue, measurement pending

The queue is live and owns zero chips while waiting for the complete six-endpoint
fixed-two MAIN and its audits. It then measures the frozen primary prediction
pair:fixed2 at2.0098B targets and One at2.4206B matched nominal training work.
No prediction-cost result is available yet.

Use the actual trained weights,remove the training-only head,and compare native
predictions against independent pinned upstream CPU FP32 references on64 fixed
authentic TRAIN windows. Time only the supported operating points:One depth1
and fixed depth2. Two rounds explicitly exchange the eight-chip halves,with
100 warm and100 timed synchronized forwards per rank. Preserve numerical gates,
checkpoint identities,weights and the exact trained endpoint updates.

The separate post-MAIN stage has a30-minute cap including CPU preparation;
it makes zero training updates and runs regardless of the quality ranking.
The reused conservative measured-training bound is1566.03 seconds,even though
the older bound allowed all three recurrent exits and the new measurement uses
only the trained depth2. Failed work and CPU-only tails remain charged.

281 CPU tests pass,including actual half-lease order,the reused-first-phase
restart case and failure before admitting a sibling. The native benchmark is
the qualified previous prefill driver with a dynamic exact completed-update
check and a guard against using a stable unfinished checkpoint.

This is fixed1024-window,preloaded full-logits prefill at batch1 per core.
It does not report KV-cached decoding or end-to-end serving. No TEST,reserve,
terminal alpha or scientific-quality claim follows from this registration.

[Protocol](../../docs/FIXED_TWO_PREFILL_COST_20261025.md) ·
[Frozen queue](queue-plan.json) · [CPU tests](cpu-tests.json).
