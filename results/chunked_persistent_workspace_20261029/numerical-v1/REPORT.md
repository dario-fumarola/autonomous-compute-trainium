# Initial qualification stops during checkpoint handling

All four actual-shape cold gradient checks pass. All 100 registered warm updates
also complete: mean seconds are 0.297368 full ordinary, 0.323220 chunked ordinary,
0.388513 generated slot and 0.389913 constant slot. Generated/full-ordinary is
1.306506. These satisfy the original throughput ceilings.

Compilation consumes most of the phase. The controller stops at its immutable
20-minute Native cap during update-103 state capture. The complete rank-zero
checkpoint and 60 of 64 replica receipts exist. This is not a completed replica,
restart or mature-gradient qualification. No scientific fit is admitted.

The failed attempt consumes 5.340969 assigned chip-hours. Its separate 5.364297
whole-instance opportunity chip-hours overlap assigned work and must not be
added. No failed work is omitted. A separately registered recovery retains the
original absolute 20:46:23 UTC deadline and all numerical/resource gates.

[Failure](report.json) · [Completed partial measurements](partial-measurements.json) ·
[Costs](execution-costs.json) · [Recovery limits](../../../docs/CHUNKED_PERSISTENT_QUALIFICATION_RECOVERY_20261029.md).
