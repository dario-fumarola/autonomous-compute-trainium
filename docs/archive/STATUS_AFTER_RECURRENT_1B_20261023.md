# Status — 2026-10-23, 13:04 UTC

**The three-seed 1B recurrent study is complete. Ordinary pretraining wins.**
At 1,004,887,590 real source targets/run, ordinary mean NLL is **2.519907** and
fixed recurrence **2.550448**. Ordinary-minus-recurrent gain is **-0.03054**,
paired-seed 95% interval **[-0.03220, -0.02888]**. Every seed and every endpoint
domain mean favors ordinary. The separately annealed equal-FLOP comparison
also favors ordinary: 2.519907 versus 2.695650. The declared 0.01-nat useful
gain is ruled out for this recipe/exposure. [Report and curves](results/recurrent_learning_20261023/REPORT.md).

This is 150M random-initialization pretraining on natural web/math/code only.
The shared recurrent path uses 1.6919× estimated training matmul FLOPs at equal
tokens. Paired initialization, source order/exposure, parameter count and global
batch match the explicitly reused ordinary controls. Each architecture has its
own equal-budget LR selection. Full 64-rank numerical, replica-equality and
optimizer recovery checks pass. A separately implemented FP32 CPU audit scores
all six complete endpoint panels and confirms native NLL within 0.0000326 nats.
Forty-two CPU tests pass. No generated research data, teacher or RL is used.

The primary controller ended at **07:05:38 UTC**. There are **no active training
or evaluation jobs**. The follow-on qualification never launched and the node
has been idle afterward; the execution did not use the full intended window.
The frozen 11:12 campaign deadline passed. The 3B continuation, fused-AdamW
hardware qualification, inference-unroll matrix and TEST confirmation remain
**unexecuted**. Their scripts are drafts, not qualified results. A 3.25B-target
append-only natural corpus is prepared, with the original packed prefix,
DEV/TEST and split/duplicate controls preserved. TEST remains sealed.

The [existing 1.7B checkpoint probe](results/transfer_probe_20261023/REPORT.md)
is complete: initial trace-versus-raw class-NLL gains disappear under identical
linear-readout adaptation. This is a frozen-feature probe; full backbone
adaptation remains untested. The [pause-locality readout](results/pause_locality_20261023/REPORT.md)
finds ordinary wins in every bin. The original [1B pause result](results/learning_curve_20261022/REPORT.md)
and all prior results are preserved. The [failed normalization screen](results/recurrent_screen_20261023/REPORT.md)
is documented separately; its scientific main fits never launched.

The recurrent campaign's closed execution receipts include qualification,
failed/stopped screens, tuning and readouts: 47.47 assigned chip-hours and
3.54 hours in the union of accelerator-active intervals. Reused ordinary
training is not charged again. CPU work and idle reservation time are separate.

All six complete terminal model/Adam/schedule/cursor/RNG states now have
hash-verified private copies on the instance's root EBS volume, with additional
NVMe copies. These are on-instance backups, not off-instance recovery. An
inactive legacy compilation cache was relocated losslessly to the unused
fourth NVMe, retaining its original path as a symlink. Original scientific
states, host drivers and XLA environment are preserved. Storage and memory
reserves pass. [OSS notes/reproducers](docs/OSS_NOTES.md) remain local/public as
appropriate; nothing has been submitted upstream.
