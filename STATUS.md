# Status — 2026-10-23, 19:05 UTC

**All scientific fits and independent endpoint audits are complete. The frozen
12-model TEST confirmation is running on all 16 chips.** Checkpoint identities,
trained depths, evaluator and analysis hashes were fixed before TEST access.
No model, LR, endpoint or inference-depth choice uses the confirmation scores.
[Frozen release](results/frozen_test_20261023/release.json).

At 1,004,887,590 real targets, the three-seed DEV means are:

| Model | Unique parameters | DEV NLL |
| --- | ---: | ---: |
| Ordinary | 150,436,608 | 2.519907 |
| Shared two-pass | 150,436,608 | 2.550448 |
| Untied two-pass | 251,124,480 | 2.490501 |

Untied improves over ordinary in every seed. The combined mean gain is
**0.029406 [0.013225, 0.045586]**. This includes discovery seed23. The two
fresh-seed interval is much wider, **[-0.052818, 0.112211]**, and inconclusive
at the 0.01-nat threshold. This is a larger-network data advantage, not a
parameter-matched or compute-efficiency result. All seeds use the selected
3e-4 recipe without new tuning. [Completed replication](results/confirmation_followup_20261023/REPORT.md).

The fully annealed ordinary resource control scores **2.421944** at
1,699,926,288 targets, versus **2.489247 untied** at 1,004,887,590. Ordinary
leads by **0.067303 [0.062260, 0.072733]**, conditional on seed23. The two
estimated training matmul budgets differ by only 0.0133%; vocabulary projection
is included. The equal-data untied gain does not survive this resource control.
Recorded retained training paths take 42.40 minutes ordinary and 39.61 minutes
untied23, with five versus two process segments. The ordinary parent-prefix
time is reconstructed from the original checkpoint timestamp. These are not
exactly wall-time-matched endpoints. [Detailed resources](results/confirmation_followup_20261023/RESOURCES.md).

The unchanged shared seed23 continuation at 3,014,637,826 targets scores
**2.336800**, versus **2.336914 ordinary**. Gain is **0.000114
[-0.002381, 0.002077]** conditional on this one pair. The 2.713B stable gain
of 0.012133 does not survive endpoint annealing. Longer exposure removes the
early deficit in this seed, without establishing the declared useful gain.
The original three-seed 1B negative remains unchanged.
[Longer-exposure curve](results/sharing_exposure_20261023/REPORT.md) ·
[Figure](results/sharing_exposure_20261023/learning-curve-seed23.png).

All new independent full-panel FP32 audits pass; the largest new CPU/native
NLL difference is 0.00004176 nats. Earlier six-endpoint audits also passed.
The three generic sharded evaluator templates reproduce scalar NLL exactly.
Their 64-rank startup/verification costs are recorded; they do not establish a
small-panel latency speedup. Nine new full-state model/Adam/RNG/cursor copies
pass checksum and serialized-state readback on existing EBS. Copies are
on-instance only; raw weights, optimizer state and source contents stay private.
The root filesystem retains approximately 39 GiB free, above the 35 GiB floor.

The first campaign and follow-up are executed by persistent bounded controllers;
a companion liveness record tracks controller and native workers independently
of stage-boundary snapshots. The realized follow-up deadline is **21:00:29 UTC**.
The performance controller waits until all scientific and TEST scoring closes.
It then attempts bounded unscaled fused-AdamW, two concurrent half-node jobs,
100 synchronized warm updates and complete optimizer recovery. No scientific
model uses that candidate optimizer. Optional AMP-metadata operator checks
remain separate. No upstream submission has been made.

Current CPU suite: **53 tests pass**. Training/evaluation use frozen natural
web, mathematical web and repository code, without a generated research corpus,
teacher, RL, halting or another architecture in this cycle. Host drivers and
XLA are preserved. All previous results and original protocols remain intact.

[Mechanism diagnostic](results/recurrent_mechanism_20261023/REPORT.md) ·
[Original primary](results/recurrent_learning_20261023/REPORT.md) ·
[Follow-up protocol](docs/CONFIRMATION_FOLLOWUP_20261023.md) ·
[Prior status](docs/archive/STATUS_BEFORE_TEST_READOUT_20261023.md).
