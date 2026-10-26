# Restore protected inode headroom without losing source or state

The first baseline numerical stage stopped with a paired-stop wrapper. Its
preserved filesystem reproduces the inode guard:8,123 free inodes versus the
protected10,000 minimum. The original top receipt did not retain the initiating
exception; source,byte-space and memory checks otherwise pass.

A dry run verifies30,000 CPython cache files against their retained recorded
sources. The bounded removal restores free inodes from8,121 to38,258 in11.318207
seconds, removing327,629,565 regenerable cache bytes across69 closed snapshots.
All28,149 corresponding source files and launch/closure receipts verify unchanged.
No source,model,optimizer,data or native compiler artifact is removed. Native
owners are absent throughout. These caches can be regenerated from the sources.

Recovery now reserves inode space for all remaining source copies, native
bytecode and per-rank artifacts before launch, in addition to the10,000 guard.
Two safety tests pass; the full repository suite has316 passing tests.

[Receipt](report.json) · [Dry run](dry-run.json) ·
[Maintenance code](../../scripts/reclaim_closed_snapshot_bytecode.py).
