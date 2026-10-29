# Sparse persistent screen stopped before quality selection

The generated-workspace LR 0.003 attempt encounters a non-finite global
gradient norm after update 80, its last persisted telemetry point. Gradient
clipping stops before an invalid update. The exact failing update is not
recorded. No endpoint or recovery checkpoint exists for this attempt. The
supervisor records no OOM, resource failure or cleanup error.

Five earlier fits complete their full schedules and mature audits: all three
ordinary LRs, generated LR 0.0003 and generated LR 0.001. Their preserved
states remain qualified. The constant-slot opportunities had not started.
No held-out scoring or LR selection occurred in this screen.

**Status: STOPPED. Original all-nine-complete advancement condition: UNMET.**
The failed learning rate is not retried, replaced or assigned an NLL. This
failure does not establish a held-out quality ranking or a universal latent
computation negative. The registered position diagnostic is not admitted,
because its prerequisite requires a complete original screen.

This attempt uses 24.737274 assigned chip-hours, including 2.172491 for the
failed fit. The preceding qualification histories add 15.304764 chip-hours.
[All execution costs](execution-costs.json) · [Original report](report.json).

A [separate bounded completion](../../../docs/CHUNKED_CONTROL_COMPLETION_20261029.md)
retains the five audited histories and runs only the original three unstarted
constant-slot opportunities. The original deadline, scientific recipes and
failed record stay unchanged. Its later descriptive comparison cannot repair
this registration or claim a bracketed generated LR optimum. Both fresh source
reserves remain sealed; terminal alpha spent is zero.
