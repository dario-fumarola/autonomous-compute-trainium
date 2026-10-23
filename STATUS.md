# Status — 2026-10-23, 13:47 UTC

**A new bounded mechanism-diagnosis campaign is active on all 16 chips.**
The original three-seed 1B result stays negative: ordinary 2.519907 NLL,
shared recurrence 2.550448, gain -0.03054 [-0.03220,-0.02888].
[Completed primary](results/recurrent_learning_20261023/REPORT.md).

The full same-weight inference matrix has completed all 15 cells.
Every completed trained-path calibration reproduces its original score.
The recurrent checkpoints already score about 3.04–3.06 with one pass versus
2.55 with their trained two-pass path. Applying two passes to ordinary
checkpoints worsens NLL to about2.64. Four passes worsen recurrent NLL to about 2.69.
This establishes trained-path dependence, not better pretraining or autonomous
reasoning. It narrows the failure from unused computation to inferior overall
learning despite useful dependence on the second pass.

The fixed natural-prefix CPU probe is complete in matching isolated-document
TRAIN/DEV formats, for all three pairs. Recurrence loses to ordinary on both
small panels. State variation remains substantial across token positions.
On three frozen TRAIN prefixes, two application-specific core gradients have
positive aggregate cosine 0.227; this is observational, not a causal sharing
test. In saved norms after 100M, no recurrent sample out of 1,053 exceeds the
clip threshold. These are sampled logs, not every update. Persistent clipping
and gross state collapse are not supported by these checks.

A detached queue, with atomic stage receipts, heartbeat and an independent
hard-deadline guard, will continue through the untied sharing control and
one seed 23 3B ordinary/recurrent continuation. New window ends 19:30 UTC.
Each fit requires native numerical qualification, full optimizer recovery and
a complete measured budget. The untied path has passed actual-wrapper CPU
function/gradient checks; full-size initial functions are bit-identical to
the shared path. Its extra parameters are explicitly counted. Native untied qualification is now running across all 16 chips; no main fit has launched. TEST stays sealed; no generated research data, teacher or RL.
[New protocol](docs/MECHANISM_DIAGNOSIS_20261023.md) · [Mechanism readout](results/recurrent_mechanism_20261023/REPORT.md).

The 3B continuation keeps the original 64-rank plain-AdamW topology and full
stable 900M state, including moments, cursor and RNG. It does not depend on
qualifying a new fused optimizer. The append-only 3.25B natural corpus is ready.
The old 11:12 campaign remains closed; this is a separate execution window.
The previous orchestration gap and unexecuted follow-ons remain recorded in
[the archived status](docs/archive/STATUS_AFTER_RECURRENT_1B_20261023.md).

All previous experiments, source snapshots and private recovery states are
preserved. Host drivers/XLA are unchanged, and reserve guards remain active.
The queue recovery/failure integration test passes. No upstream OSS submission
has been made.
