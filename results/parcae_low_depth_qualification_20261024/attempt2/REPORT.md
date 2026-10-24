# Low-depth engine revision2: queued behind numerical gates

This revision stopped at its failed recovery prerequisite with zero native
phases. [Revision3](../attempt3/REPORT.md) retains its model/trainer/auditor and
changes only admission, host caps and CPU-arena reclamation.

Same prepared three-arm initializations, exact natural source/depth/noise tables,
context1024 and accumulation8/16 cases. The explicitly changed optimizer uses
FP32 polar arithmetic and eager Muon; every model has the same optimizer policy.
The suspended BF16-dependent engine and its original39-file freeze are retained.

The new43-file freeze includes training, recovery and independent first-update
and trained-gradient audit implementations. Each qualified precision/case gets
20 warmup plus100 measured warm updates, an uninterrupted121st update and a
fresh-process replay. Independent reference checks target the actual first
optimizer step and the complete objective after120 updates. No scientific fit,
held-out scoring, fresh confirmation or automatic large-run launch follows.

Latest native phase start04:00UTC October25,90-minute complete execution cap,
30minutes per native phase/CPU audit. Both eight-chip halves are used after all
prerequisites pass; waiting reserves zero chips. Preserve2% numerical gates and
complete exact state on load; do not promise bit-identical future trajectories.

[Plan](native-plan.json) · [Live status](native-status.json) ·
[Original CPU preparation](../cpu-report.json) ·
[Execution amendment](../../../docs/PARCAE_PRECISE_EXECUTION_20261024.md) ·
[Unstarted registration failure](unstarted-registration/failure.json).
