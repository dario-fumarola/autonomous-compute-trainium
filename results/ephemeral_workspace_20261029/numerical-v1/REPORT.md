# Disposable ephemeral-state qualification passes

The51,747,328-parameter source-KV reuse graph passes on all16chips/64ranks with
2×512-position windows per rank. Cold independent gradient checks pass;
the complete128-window mature audit has maximum family relativeL2 error
1.08686864e-06 and loss difference1.71363354e-06. No tolerance changed.

One hundred warm updates average0.191977seconds. A fresh
process restores the complete update103 model/optimizer/RNG state and reproduces
update104 exactly, with identical replicas. This is qualification timing,
not scientific prediction quality or an inference-throughput measurement.

The two Native attempts use1.402411assigned chip-hours; the independent CPU
audit takes31.003seconds. The prior admission stopped at the
inode projection before any Native worker. Its separate recovery retains the
original cutoff after exact preservation of old compiler temporaries.

All qualification states are excluded from scientific initialization. The
prepared three-LR trained comparison still requires both new replication seeds
to pass. No held-out source was read by this qualification, no fresh reserve
opened and no terminal alpha spent.

[Complete numerical receipt](report.json) · [All execution costs](execution-costs.json) ·
[Admission recovery](../admission-recovery-v1/plan.json).
