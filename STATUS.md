# Status — 2026-10-23, 05:35 UTC

**Three-seed fixed recurrence is running on all 16 chips.** At 301,465,908 real
source targets/run, ordinary mean NLL is 2.97414 and recurrent mean NLL is
2.99425: gain -0.02011 [-0.03381, -0.00642]. The deficit narrows from 0.239 at
100M. These are stable-trunk development curves; the annealed 1B primary and
independently annealed equal-FLOP endpoints are still pending. TEST is sealed.
[Current curves](results/recurrent_learning_20261023/REPORT.md).

The 150M pre-norm paths have identical paired initial weights, parameters,
source order and global batch. The ordinary controls are reused explicitly
from the completed study. CPU outputs/full gradients prove exact single-pass
equivalence. All three recurrent LR trials pass; 3e-4 is frozen versus the
ordinary reference's independently selected 1e-3. Full 64-rank gradient,
replica and optimizer-recovery checks pass. The measured warm recurrent rate
is 604k real targets/sec. Forty-one CPU tests pass.

The active [protocol](docs/PRELN_RECURRENT_PROTOCOL.md) and
[remaining-time allocation](docs/LONGER_EXPOSURE_ALLOCATION.md) fix the claims,
controls and 11:12 UTC campaign deadline. A natural append-only extension is
ready: 3,250,005,799 targets, all 912,935 original document identities recovered
for global duplicate screening, zero group overlap and unchanged original
packed prefix/DEV/TEST. Longer training remains gated on native recovery and
measured complete-run time. No generated research data, teacher or RL is used.

An independent, separate-projection FP32 CPU implementation passes full
forward/gradient checks. It scores the complete curve panel: all three ordinary
endpoint NLLs agree with native to within 0.000022 nats. Recurrent endpoints
will be checked as they finish. A live runtime sample shows all 64 logical
cores active, zero execution errors and roughly 65% runtime busy occupancy;
this is not peak-FLOP utilization or a campaign average.

The [existing 1.7B checkpoint probe](results/transfer_probe_20261023/REPORT.md)
is complete: initial trace-versus-raw class-NLL gains disappear under identical
linear-readout adaptation. Both final intervals exclude a 0.01-nat gain for
this limited probe. Full backbone adaptation remains untested. The
[pause-locality readout](results/pause_locality_20261023/REPORT.md) is complete:
ordinary wins in every bin. The original [1B pause result](results/learning_curve_20261022/REPORT.md)
and all states remain preserved. [Previous complete status](docs/archive/STATUS_AFTER_PAUSE_20261022.md).

Failures remain recorded: the initial sandwich recipe's six LR trials collapse
to nearly context-independent prediction and its 1B fits never launch; a BF16
feature-batching check fails before trained models are scored; download-only
preparation is stopped for a caught copy-boundary bug; one independent CPU
gradient-export test needs correction. No primary scientific state is restarted.

New scratch and copied runtime caches use the previously unused fourth local
NVMe; original copies and filesystems remain intact. Host drivers and the
preserved XLA environment are unchanged. Full states stay private, with reserve
guards active. This is on-instance storage, not an off-instance backup. Local
[OSS notes/reproducers](docs/OSS_NOTES.md) are retained; nothing is submitted upstream.
