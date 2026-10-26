# Fixed-depth controls and a second development seed

The completed seed-5231 fixed-two model has a 0.012016-nat advantage over
One at matched nominal matrix work. Its conditional source-group interval
overlaps the 0.01 useful-effect threshold. This is development evidence, not
confirmation. Complete the missing fixed-two cheap control and repeat the
comparison with the already prepared, unused development seed 17293.

Freeze five new cooled endpoints before native launch:

| Seed | Model | Updates | Purpose |
| --- | --- | ---: | --- |
| 5231 | Two mixes, one core | 4,594 | Missing cheap work control |
| 17293 | Fixed two cores | 3,840 | Equal-data recurrent candidate |
| 17293 | One pass | 3,840 | Equal-data control |
| 17293 | One pass | 4,625 | Matched-work control |
| 17293 | Two mixes, one core | 4,594 | Matched-work cheap control |

Use the same original random initializer within each seed, source order,
parameter inventory, qualified batch-one numerical stack and next-token loss.
No auxiliary objective, architecture change, new rate or per-seed tuning.
The three independently selected rates are all 0.032, each an upper-boundary
selection in its original three-rate grid. The completed cheap TUNE and all
its prefix and endpoint gradient gates are prerequisites, not scientific
outcomes of this stage. No new operator needs qualification here.

Every endpoint completes its own absolute warmup/stable/cosine schedule,
with 50M-target warmup, cooldown over its final half and zero final rate.
One's two seed-17293 histories share only their untouched stable 1,920-update
prefix. Never extend an annealed checkpoint. Six native fit segments make
19,573 actual updates; common prefixes are charged once physically and to
both complete conceptual model histories. Save only the six necessary
segment-end states. All five final checkpoints require complete 512-window
native/independent-CPU gradient agreement, full-state readback and equal
replica states. The One branch also requires exact restored model, optimizer
and RNG state. This does not establish bit-exact future restart trajectories.

Use exact integer work floors against 1,272,689,958,300,549,120 operations
for fixed2. Count executed attention, vocabulary and common zero-weight
auxiliary heads, backward and 32 replicated optimizer matrices. This semantic
ledger excludes physical padding, pointwise work and communications; record
actual chip-time separately. Do not reprice completed fits as lean models.

Admission requires actual 100-update timings for One, fixed2 and cheap,
the complete stage including independent CPU tails fitting a six-hour cap,
at least 70 GiB free on NVMe, and the protected 35 GiB filesystem / 512 GiB
available-memory / 10,000-inode reserves. Bound each native phase to 9,000
seconds and CPU phase to 1,200 seconds within the original stage deadline.
Use both eight-chip halves when ready. Cleanup cancels only owned workers.
Resume validates exact saved paths and child-plan hashes, reuses successful
closures and keeps the original clock. Partial failed phases require explicit
recovery registration; neither errors nor unfavorable scores extend scope.

Score every endpoint on the same consumed DEV cohort, with fixed 60/20/20
web/math/code weights. Reuse the completed seed-5231 One and recurrent scores
without rescoring or refitting them. Report, for each of both seeds, recurrence
versus equal-data One, matched-work One and matched-work cheap. Report both
control contrasts and their descriptive two-seed averages; never select away
the stronger control. Source-group intervals remain conditional on fitted
weights. Two development seeds do not supply terminal training-seed inference.
Keep all failed and unfavorable outcomes. Do not infer a refinement mechanism
from deleting a pass that fixed-depth training did not support.

This stage accesses no TEST or fresh reserve and spends zero terminal alpha.
It cannot complete the research goal. Stronger qualified lean training,
measured inference costs, at least five fresh paired confirmation seeds and
fresh source groups remain requirements for a terminal positive claim.
