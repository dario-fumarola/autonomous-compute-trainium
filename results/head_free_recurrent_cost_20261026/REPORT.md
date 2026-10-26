# Read-only head-free recurrent cost check

The existing fixed-two matrix ledger charges331,429,676,640,768 operations per update. Removing the zero-weight future head leaves326,673,335,904,768,a1.435098% reduction. The original1,272,689,958,300,549,120 budget admits3,895 complete head-free updates rather than3,840. Verified source counters give2,038,547,734 versus2,009,762,680 authentic targets. The next complete update exceeds the budget. [Exact arithmetic and source hashes](report.json).

This removes24,608,000 training-only parameters while leaving141,963,840 deployed parameters. It is a prospective ledger calculation,not a measured hardware saving or prediction of the NLL effect. The nominal matrix accounting does not include pointwise operations,compile costs or memory traffic,and this check does not establish which work the compiler actually prunes.

No old result is retroactively credited. The active final-quarter run retains its registered head and complete work ledger. A future head-free recurrent candidate requires its own prediction/gradient/state qualification,from-random training,complete cooldown and strongest ordinary8 comparison. Step conditioning and deterministic-state initialization remain separate hypotheses,not identified defects. No qualification or scientific training is admitted by this calculation.
