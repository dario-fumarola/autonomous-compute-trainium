# Status — 2026-10-23, 16:23 UTC

**The unchanged seed-23 3B continuation is training on all 16 chips.**
Both paths have completed the stable 2.010B-target checkpoint: ordinary
2.463624 NLL, shared recurrence 2.462959. This near tie is one seed at stable
LR, not the annealed endpoint or a replicated efficiency result. Ordinary has
also finished training to the 2.713B stable milestone; shared is following.
The paired annealed 3.015B endpoints and independent FP32 audits remain pending.

The original three-seed 1B result is preserved: ordinary **2.519907**, shared
**2.550448**, ordinary-minus-shared gain **-0.03054 [-0.03220,-0.02888]**.
The seed-23 untied control completes 1.005B targets at **2.489247**, versus
**2.518071 ordinary** and **2.549316 shared**. Its extra 100.7M parameters are
intentional. This positive is a sharing-plus-capacity control, not a
parameter-matched or total-cost-matched computation advantage.

The fifteen-cell same-weight inference matrix is complete: shared checkpoints
need their trained second pass (mean NLL 3.050335 with one, 2.550448 with two).
Four passes worsen NLL to 2.693321. These results establish trained-path
dependence, not a reusable operation that improves every additional iteration.
State, clipping, gradient and document-boundary diagnostics remain in the
[mechanism report](results/recurrent_mechanism_20261023/REPORT.md). Untied
weight-projection and spectrum readouts are exploratory perturbations, not
retrained sharing controls or evidence that small adaptations cannot work.

A **separate bounded confirmation follow-up** is now declared, before either
new fit and before TEST access: untied seeds **47 and 71**, using the already
selected LR 3e-4 and exact existing recipe; and ordinary seed 23 at **6,496
updates**, approximately 1.70B targets, to match the untied endpoint's estimated
training matmul work. That control forks the original stable 900M full state,
keeps all Adam moments and completes its own decay. No new architecture,
learning-rate search, teacher, synthetic research corpus, RL or halting.
[Follow-up protocol](docs/CONFIRMATION_FOLLOWUP_20261023.md).

The detached follow-up coordinator waits for the current campaign, independent
endpoint audits and three trained-path DEV evaluator calibrations. It then
requires a conservative measured complete budget with protected confirmation
and reporting reserves. The follow-up lasts at most four hours, with an
absolute ceiling of **23:30 UTC**. Both fresh seeds have already passed full-size
initial-weight/function equality against their paired original controls.

**TEST remains sealed.** The old nine-model postprocessing waiter was explicitly
superseded before any TEST access. The amended scope fixes all twelve terminal
models before release. The original three-seed comparison stays primary;
combined untied intervals include discovery seed23, while fresh seeds47/71 are
reported separately with df=1. Resource matching is one pair, with conditional
source-group intervals. All domains and declared comparisons are retained.
[Confirmation protocol](docs/FROZEN_TEST_CONFIRMATION.md).

Private full-state backup/readback checks run on CPU. Every native stage keeps
512 GiB host memory and 35 GiB free filesystem reserves. An inactive package
cache was copied byte-for-byte to instance NVMe, checksum verified, and its
original path retained by symlink to make room for recovery states; installed
environments, drivers and XLA remain unchanged. Copies are on-instance only.
The post-science fused-optimizer qualification waits until every scientific
fit and TEST readout is closed; it cannot change completed models. No upstream
OSS submission has been made. Current CPU suite: **50 tests pass**.

All earlier results and original protocols remain preserved, including the
[original primary](results/recurrent_learning_20261023/REPORT.md),
[sharing readout](results/sharing_exposure_20261023/REPORT.md), and
[prior campaign status](docs/archive/STATUS_BEFORE_CONFIRMATION_FOLLOWUP_20261023.md).
