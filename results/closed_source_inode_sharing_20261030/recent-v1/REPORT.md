# Recent closed source copies share storage without losing bytes

All 22,480 original source paths across 24 completed snapshots verify against
their launch manifests. Atomic replacement of 21,492 duplicate paths with
byte- and permission-identical hard links preserves every source byte and
all launch/closure receipts. Free file entries increase from 21,073 to 28,677.
Inventory takes 34.043 seconds and sharing/verification 67.685 seconds, with
zero accelerator assignments or optimizer updates.

Models, optimizer states, raw source data, live worktrees and compiler caches
are untouched. Treat historical aliases as immutable; copy them before future
debugging. The preceding whole-history inventory hit its 250,000-path bound
without mutation and remains separately recorded. Three storage-helper tests
pass. This maintenance enables full-queue admission; it does not change any
research result or relax its capacity requirements.

[Complete receipt](report.json) · [Inventory](inventory.json) ·
[Registration](registration.json)
