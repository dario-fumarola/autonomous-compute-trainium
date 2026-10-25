# Compute each Muon matrix group once — CPU candidate only

The qualified optimizer repeats the same batched polar calculation on every
DDP replica. Its actual100-step one-pass measurement charges26.092 trillion
optimizer matmul operations per global step,versus249.056 trillion neural/head
operations. The repeated work is counted honestly in existing comparisons;
this preparation changes none of their budgets or outcomes.

A separate candidate assigns each independent shape group to an owner rank.
Owners calculate their groups before the common broadcast sequence,then
broadcast the complete weights,momentum and variance. Other replicas keep
identical optimizer state and the original checkpoint schema. This distributes
computation;it does not shard stored state. Owners are spread across the group
when there are more ranks than matrix shapes. The physical execution is named
separately from the unchanged mathematical recipe.

Two real Gloo processes match the qualified unsharded FP32 update bit for bit,
including every moment,zero gradients,changing schedules and state restoration.
All249 CPU tests pass. Fixtures are numerical software checks,not research
examples. No native optimizer,training run or throughput improvement is claimed.

The [SDK2.33 native release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/release-notes/components/torchneuron.html)
list broadcast support. That capability statement is not numerical or
performance qualification. Extra communication could erase any benefit.
A future attempt must measure actual latency,full-state recovery and independent
CPU agreement,then full-training throughput,under its own finite budget. Keep
it behind completed scientific work and reject it if it is slower. Do not
change the running trainer or submit anything upstream without human review.
[CPU evidence](../results/owner_broadcast_muon_preparation_20261025/cpu-report.json).

## Completed native performance screens

The separate matrix-fixture screen completes120 updates per variant,including
100 warm timed updates:on all actual Muon matrix shapes,replicated takes7.074s
and owner/broadcast4.191s. First/final weights and moments match bit for bit
across variants and all32 replicas. This is one concurrent hardware pair,not
full-training speed or independent CPU/recovery qualification.
[Matrix-only evidence](../results/owner_muon_fixture_native_20261025/report.json).

The subsequent full one-pass training screen completes121 qualification
updates per variant with100 warm measurements. Replicated takes110.170s
versus106.911s owner/broadcast,only2.96% lower time. Final model/optimizer
states match exactly. This fails the predeclared5% performance screen. Close
the proposal at this stage;no further adoption work or scientific use follows.
[Full-training evidence](../results/owner_full_training_screen_20261025/report.json).
